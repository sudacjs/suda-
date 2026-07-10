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
def load_and_preprocess_data():
    print("Loading datasets from Hugging Face...")
    try:
        df_10m = load_dataset("Antajitters/WindSpeed_10m", split="train").to_pandas()
        df_50m = load_dataset("Antajitters/WindSpeed_50m", split="train").to_pandas()
        df_100m = load_dataset("Antajitters/WindSpeed_100m", split="train").to_pandas()

        # 动态获取时间列的名字，保证鲁棒性
        time_col = df_10m.columns[0]

        # 将三张表按时间列进行拼接
        df = df_10m.merge(df_50m, on=time_col, suffixes=('_10', '_50'))
        df = df.merge(df_100m, on=time_col)

        # 统一设置索引并进行缺失值处理
        df.set_index(time_col, inplace=True)
        df = df.interpolate(method='linear').bfill()

        print("数据加载与拼接完成。")
        return df

    except Exception as e:
        print(f"数据加载出错: {e}")
        return None
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
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, model_name="model"):
    # 损失函数使用均方误差
    criterion = nn.MSELoss()

    # 【核心修改】引入 weight_decay=1e-3
    # 这会给模型的权重添加一个 L2 惩罚项，强制模型倾向于学习更简单、更平滑的规律，
    # 从而有效抑制对噪声的“死记硬背”，提升长线预测的泛化能力
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_loss = float('inf')

    # 确保保存模型的文件夹存在
    os.makedirs('saved_models', exist_ok=True)

    for epoch in range(epochs):
        # --- 训练阶段 ---
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

        # --- 验证阶段 ---
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

        # 仅保存验证集 Loss 最低的模型，防止模型过拟合到训练集后期
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"saved_models/{model_name}.pth")

    print(f"训练完成！最佳模型已保存至 saved_models/{model_name}.pth")


def evaluate_and_plot(model, test_loader, target_scaler, model_name="model", task_type="single"):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            # 反归一化，还原成真实风速数值
            all_preds.append(target_scaler.inverse_transform(outputs.cpu().numpy()))
            all_trues.append(target_scaler.inverse_transform(batch_y.cpu().numpy()))

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    # 计算评估指标
    mse = np.mean((all_trues - all_preds) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_trues - all_preds))
    # 计算 R2
    ss_res = np.sum((all_trues - all_preds) ** 2)
    ss_tot = np.sum((all_trues - np.mean(all_trues)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\n[{model_name} - {task_type}] 指标评估:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 绘图部分
    plt.figure(figsize=(10, 5))
    plt.plot(all_trues[:100, 0], label='True Value', color='blue')
    plt.plot(all_preds[:100, 0], label='Prediction', color='red', linestyle='--')
    plt.title(f"{model_name} Prediction Comparison ({task_type})")
    plt.legend()
    plt.savefig(f"{model_name}_{task_type}_result.png")
    plt.show()
    plt.close()
# ==========================================
# ==========================================
# 5. 主执行流程
# ==========================================
if __name__ == "__main__":
    df = load_and_preprocess_data()

    # 因为我们要拯救 16 小时的预测，所以确保这里设为了 multi_B
    CURRENT_TASK = 'multi_B'

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


        train_model(model, train_loader, val_loader, epochs=5, lr=0.0005, model_name=f"{name}_{CURRENT_TASK}")

        # 从 saved_models 文件夹加载模型
        model.load_state_dict(torch.load(f"saved_models/{name}_{CURRENT_TASK}.pth"))
        evaluate_and_plot(model, test_loader, target_scaler, model_name=name, task_type=CURRENT_TASK)

    print("\n所有任务执行完毕。请检查生成的 saved_models 文件夹和 .png 可视化图表。")