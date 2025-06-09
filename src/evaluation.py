import os
import torch
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('/workspace/mamba')
from mamba_ssm import Mamba

# === CONFIG ===
VERSION = "v7"
MODEL_PATH = f"best_mamba_v7_model.pt"
EVAL_DIR = f"mamba_{VERSION}_UE3/evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

SEQ_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_col = 'UE3-Jitter'
input_cols = ['UE3: web-rtc', 'UE3: sipp', 'UE3: web-server', 'UE3-CQI']

# === LOAD DATA ===
df = pd.read_csv("../data.csv")
df.columns = df.columns.str.strip()
df['UE3-Jitter_lag1'] = df[target_col].shift(1).fillna(method="bfill")

# Feature engineering (must match training)
for col in input_cols + ['UE3-Jitter_lag1']:
    df[f"{col}_delta"] = df[col].diff().fillna(0)
    df[f"{col}_rollmean"] = df[col].rolling(window=5, min_periods=1).mean()
    df[f"{col}_rollstd"] = df[col].rolling(window=5, min_periods=1).std().fillna(0)

engineered_features = []
for col in input_cols + ['UE3-Jitter_lag1']:
    engineered_features += [col, f"{col}_delta", f"{col}_rollmean", f"{col}_rollstd"]

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()
df[engineered_features] = scaler_X.fit_transform(df[engineered_features])
df[[target_col]] = scaler_y.fit_transform(df[[target_col]])

# Sequence builder
def create_sequences(df, seq_len, target_col):
    X, y = [], []
    for i in range(len(df) - seq_len - 1):
        seq_x = df.iloc[i:i+seq_len][engineered_features].values
        seq_y = df.iloc[i+seq_len+1][target_col]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = create_sequences(df, SEQ_LEN, target_col)
X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# === MODEL (same as training) ===
class MambaV7(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.mamba1 = Mamba(hidden_dim)
        self.mamba2 = Mamba(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        skip_out = self.skip(x[:, -1, :])
        x = self.encoder(x)
        x = self.mamba1(x)
        x = self.mamba2(x)
        x = self.norm(x)
        x = x[:, -1, :]
        return self.head(x) + skip_out

# Load model
model = MambaV7(input_dim=X.shape[2]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
torch.cuda.reset_peak_memory_stats()
start_time = time.time()


# Inference
with torch.no_grad():
    y_pred = model(X_tensor).cpu().numpy()

end_time = time.time()

inference_time = end_time - start_time
gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

print(f"Inference Time: {inference_time:.6f} seconds")
print(f"GPU Memory Usage: {gpu_memory_used:.2f} MB")

y_true = scaler_y.inverse_transform(y.reshape(-1, 1))
y_pred = scaler_y.inverse_transform(y_pred)

# Metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

with open(os.path.join(EVAL_DIR, "UE3-Jitter_metrics.txt"), "w") as f:
    f.write(f"UE3-Jitter Forecast (v7):\n")
    f.write(f"  MSE={mse:.6f}\n  MAE={mae:.6f}\n  R²={r2:.6f}\n")

print(f"\n📊 Evaluation - UE3:\n  MSE={mse:.6f}\n  MAE={mae:.6f}\n  R²={r2:.6f}")

# === Plots ===
residuals = y_true.flatten() - y_pred.flatten()

plt.figure(figsize=(10, 4))
plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Actual vs Predicted - UE3 Jitter (v7)")
plt.xlabel("Sample")
plt.ylabel("Jitter")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "UE3-Jitter_actual_vs_predicted.png"))
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=50, alpha=0.7, color='orange')
plt.title("Residuals Histogram - UE3 Jitter (v7)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "UE3-Jitter_residuals_hist.png"))
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.title("Predicted vs Actual - UE3 Jitter (v7)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "UE3-Jitter_scatter.png"))
plt.close()

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted - UE3 Jitter (v7)")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "UE3-Jitter_residuals_vs_predicted.png"))
plt.close()

plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals - UE3 Jitter (v7)")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "UE3-Jitter_qq_plot.png"))
plt.close()
