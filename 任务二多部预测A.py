import matplotlib

matplotlib.use('Agg')  # 强制使用无界面的后端，完美避开 tkinter 报错
import os  # 需要在代码最上方加入这行导入
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from datasets import load_dataset

warnings.filterwarnings('ignore')

# ==========================================
# 0. 基础配置
# ==========================================
# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==========================================
# 1. 数据获取与预处理
# ==========================================
def load_and_preprocess_data():
    print("Loading datasets from Hugging Face...")
    try:
        df_10m = load_dataset("Antajitters/WindSpeed_10m", split="train").to_pandas()
        df_50m = load_dataset("Antajitters/WindSpeed_50m", split="train").to_pandas()
        df_100m = load_dataset("Antajitters/WindSpeed_100m", split="train").to_pandas()

        # 动态获取时间列的名字
        time_col = df_10m.columns[0]
        print(f"成功检测到时间列名为: '{time_col}'")

        # 使用真实的时间列名进行表格合并
        df = df_10m.merge(df_50m, on=time_col, suffixes=('_10', '_50'))
        df = df.merge(df_100m, on=time_col)

        # 重命名列以保持规范
        df.columns = [col if col == time_col or '_' in col else col + '_100' for col in df.columns]
        df.rename(columns={time_col: 'Timestamp'}, inplace=True)

    except Exception as e:
        print(f"真实数据下载或处理失败，错误原因: {e}")
        print(">> 自动切换至模拟数据继续演示...")
        dates = pd.date_range(start='2016-11-19 13:50', periods=10000, freq='10min')
        df = pd.DataFrame({
            'Timestamp': dates,
            'Wind Speed': np.random.uniform(5, 15, 10000),
            'Wind Direction': np.random.uniform(0, 360, 10000),
            'Temperature': np.random.uniform(10, 25, 10000),
            'Pressure': np.random.uniform(780, 800, 10000),
            'Humidity': np.random.uniform(60, 100, 10000)
        })

    # 数据清洗：处理缺失值
    df.set_index('Timestamp', inplace=True)
    df = df.interpolate(method='linear')
    df = df.bfill()

    # 绘制数据集特征相关性图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    return df


# ==========================================
# 2. 序列窗口划分与Dataset构建
# ==========================================
class WindSpeedDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_col_idx):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col_idx = target_col_idx

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data[index: index + self.seq_len]
        y = self.data[index + self.seq_len: index + self.seq_len + self.pred_len, self.target_col_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_dataloaders(df, task_type='multi_A', batch_size=64):
    if task_type == 'single':
        seq_len, pred_len = 48, 1
    elif task_type == 'multi_A':
        seq_len, pred_len = 48, 6
    elif task_type == 'multi_B':
        seq_len, pred_len = 48, 96
    else:
        raise ValueError("Unknown task type")

    # 终极目标列定位逻辑 (忽略大小写，带保底机制)
    target_col = None
    for col in df.columns:
        if 'speed' in str(col).lower():
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[1]
        print(f"警告: 没找到带 speed 字眼的列，默认抓取第2列: '{target_col}' 作为预测目标")
    else:
        print(f"成功将预测目标锁定为真实的列名: '{target_col}'")

    target_idx = df.columns.get_loc(target_col)

    # 严格按照7:2:1划分数据集
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # 特征工程：标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_df)
    val_data = scaler.transform(val_df)
    test_data = scaler.transform(test_df)

    # 记录目标变量的缩放参数，用于后续反归一化评估
    target_scaler = StandardScaler()
    target_scaler.fit(train_df[[target_col]])

    train_dataset = WindSpeedDataset(train_data, seq_len, pred_len, target_idx)
    val_dataset = WindSpeedDataset(val_data, seq_len, pred_len, target_idx)
    test_dataset = WindSpeedDataset(test_data, seq_len, pred_len, target_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, target_scaler, seq_len, pred_len, df.shape[1]


# ==========================================
# 3. 模型定义
# ==========================================
class LinearModel(nn.Module):
    def __init__(self, input_size, seq_len, pred_len):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size * seq_len, pred_len)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_len, pred_len, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x


# ==========================================



# ==========================================
# 4. 训练与评估模块 (已修改保存路径)
# ==========================================
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, model_name="model"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    # 【新增】确保保存模型的文件夹存在
    os.makedirs('saved_models', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 【修改】保存到 saved_models 文件夹下
            torch.save(model.state_dict(), f"saved_models/{model_name}.pth")

    print(f"训练完成！最佳模型已保存至 saved_models/{model_name}.pth")


def evaluate_and_plot(model, test_loader, target_scaler, model_name, task_type):
    model.eval()
    predictions, trues = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            trues.append(batch_y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    trues = np.concatenate(trues, axis=0)

    predictions_inverse = target_scaler.inverse_transform(predictions)
    trues_inverse = target_scaler.inverse_transform(trues)

    mse = mean_squared_error(trues_inverse.flatten(), predictions_inverse.flatten())
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(trues_inverse.flatten(), predictions_inverse.flatten())
    r2 = r2_score(trues_inverse.flatten(), predictions_inverse.flatten())

    print(f"\n--- {model_name} Test Results ({task_type}) ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(trues_inverse[:200, 0], label='True Values', color='blue')
    plt.plot(predictions_inverse[:200, 0], label='Predictions', color='red', linestyle='--')
    plt.title(f'{model_name} - True vs Predicted Wind Speed ({task_type})')
    plt.xlabel('Time Steps')
    plt.ylabel('Wind Speed')
    plt.legend()
    # 图片还是保存在外面，方便你看
    plt.savefig(f'{model_name}_{task_type}_predictions.png')
    plt.close()


# ==========================================
# 5. 主执行流程
# ==========================================
if __name__ == "__main__":
    df = load_and_preprocess_data()

    # 'single' = 单步预测 | 'multi_A' = 多步预测A | 'multi_B' = 多步预测B
    CURRENT_TASK = 'multi_A'  # 你可以改成你想要跑的任务

    print(f"\nPreparing DataLoader for Task: {CURRENT_TASK}")
    train_loader, val_loader, test_loader, target_scaler, seq_len, pred_len, input_size = prepare_dataloaders(df,
                                                                                                              task_type=CURRENT_TASK)

    models = {
        "LinearRegression": LinearModel(input_size, seq_len, pred_len).to(device),
        "LSTM": LSTMModel(input_size, hidden_size=64, num_layers=2, pred_len=pred_len).to(device),
        "Transformer": TransformerModel(input_size, seq_len, pred_len).to(device)
    }

    for name, model in models.items():
        print(f"\n========== Training {name} ==========")
        train_model(model, train_loader, val_loader, epochs=10, lr=0.001, model_name=f"{name}_{CURRENT_TASK}")

        # 【修改】从 saved_models 文件夹加载模型
        model.load_state_dict(torch.load(f"saved_models/{name}_{CURRENT_TASK}.pth"))
        evaluate_and_plot(model, test_loader, target_scaler, model_name=name, task_type=CURRENT_TASK)

    print("\n所有任务执行完毕。请检查生成的 saved_models 文件夹和 .png 可视化图表。")