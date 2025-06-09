import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mamba.models.mixer_seq_simple import Mamba

# === CONFIG ===
VERSION = "v7"
OUTPUT_DIR = f"mamba_{VERSION}_UE3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD DATA ===
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

input_cols = ['UE3: web-rtc', 'UE3: sipp', 'UE3: web-server', 'UE3-CQI']
target_col = 'UE3-Jitter'

# Add lagged jitter as feature
df['UE3-Jitter_lag1'] = df[target_col].shift(1).fillna(method="bfill")

# Add deltas and rolling stats
for col in input_cols + ['UE3-Jitter_lag1']:
    df[f"{col}_delta"] = df[col].diff().fillna(0)
    df[f"{col}_rollmean"] = df[col].rolling(window=5, min_periods=1).mean()
    df[f"{col}_rollstd"] = df[col].rolling(window=5, min_periods=1).std().fillna(0)

# Final features
engineered_features = []
for col in input_cols + ['UE3-Jitter_lag1']:
    engineered_features += [col, f"{col}_delta", f"{col}_rollmean", f"{col}_rollstd"]

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()
df[engineered_features] = scaler_X.fit_transform(df[engineered_features])
df[[target_col]] = scaler_y.fit_transform(df[[target_col]])

# === SEQUENCE BUILDER ===
def create_sequences(df, seq_len, target_col):
    X, y = [], []
    for i in range(len(df) - seq_len - 1):
        seq_x = df.iloc[i:i+seq_len][engineered_features].values
        seq_y = df.iloc[i+seq_len+1][target_col]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = create_sequences(df, SEQ_LEN, target_col)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

dataset = TensorDataset(X_tensor, y_tensor)
train_len = int(0.7 * len(dataset))
val_len = int(0.2 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=1)

# === MODEL ===
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
        x = x.to(DEVICE)
        skip_out = self.skip(x[:, -1, :])
        x = self.encoder(x)
        x = self.mamba1(x)
        x = self.mamba2(x)
        x = self.norm(x)
        x = x[:, -1, :]
        return self.head(x) + skip_out

# === TRAINING SETUP ===
model = MambaV7(input_dim=X.shape[2]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.SmoothL1Loss()

best_val_loss = float("inf")
patience_counter = 0
train_losses, val_losses = [], []

# === TRAIN LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = sum(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() for xb, yb in val_loader) / len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_mamba_v7_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered.")
            break

# === LOSS CURVE ===
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve - Mamba v7 UE3")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v7_loss_curve.png"))
plt.close()

# === EVALUATION ===
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_mamba_v7_model.pt")))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb.to(DEVICE))
        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

y_true = scaler_y.inverse_transform(np.vstack(y_true))
y_pred = scaler_y.inverse_transform(np.vstack(y_pred))

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

with open(os.path.join(OUTPUT_DIR, "v7_metrics.txt"), "w") as f:
    f.write(f"UE3-Jitter Forecast (v7):\n")
    f.write(f"  MSE={mse:.6f}\n  MAE={mae:.6f}\n  R²={r2:.6f}\n")

print(f"\n📊 Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")

plt.figure(figsize=(10, 4))
plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Actual vs Predicted - UE3-Jitter (v7)")
plt.xlabel("Sample")
plt.ylabel("Jitter")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v7_actual_vs_predicted_UE3.png"))
plt.close()

