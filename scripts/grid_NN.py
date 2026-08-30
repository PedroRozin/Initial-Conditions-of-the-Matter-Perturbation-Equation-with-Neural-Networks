import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import csv
from tqdm import tqdm
import sys
import main_functions as ft
from main_functions import ImprovedRegressionNN
import os

# ===========================
# GPU Configuration
print("verifying GPU...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Memoria GPU libre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    torch.cuda.empty_cache()  # Limpiar cache

else:
    device = torch.device('cpu')
    print(" GPU no disponible, usando CPU")
print(' ')
print(f" Dispositivo seleccionado: {device}")
print("="*50)

path_folder = '/home/pedrorozin/paper_tesis2025/outputs/neural_networks/'
n = 'NN_z_approx_32_trained_dense_v2'

if os.path.exists(path_folder + n):
    raise FileExistsError(f"El directorio {path_folder}/{n} ya existe.")

if not os.path.exists(path_folder + n):
    os.makedirs(path_folder + n)


# ===========================
# 1. load data y split and scale


#select grid for training
main_path = '/home/pedrorozin/paper_tesis2025/outputs/grids/'
name_grid = 'grid_z_approx_32_validation_data'
path_grilla = f'{main_path}{name_grid}/grilla_results_{name_grid}.csv'
usecols = ['a', 'k h', 'h', 'Omega_m', 'delta_m', 'delta_prime_m', 'k_horizon']
df_grilla = pd.read_csv(
    path_grilla,
    usecols=lambda col: col in usecols,
    dtype={
        'a': 'float32',
        'k h': 'float32',
        'h': 'float32',
        'Omega_m': 'float32',
        'delta_m': 'float32',
        'delta_prime_m': 'float32',
        'k_horizon': 'float32',
    },
)
mask = (df_grilla['k h'] < 0.21) & (df_grilla['a'] < 0.035)
if 'k_horizon' in df_grilla.columns:
    mask &= df_grilla['k h'] > df_grilla['k_horizon']
df = df_grilla.loc[mask].copy()

# Features y targets

#filter features with k h <= 0.25 (no lineal regime)
features = df[["a", "k h", "h", "Omega_m"]][df['k h'] <= 0.4].to_numpy(dtype=np.float32)

targets = df[["delta_m", "delta_prime_m"]].to_numpy(dtype=np.float32)

# Split train y val (80% train, 20% val) before scaling.

X_train, X_val, y_train, y_val = train_test_split(
    features, targets, 
    test_size=0.2, 
    random_state=100, 
    shuffle=True
)

#save y_val to csv for later evaluation
validation_df = pd.DataFrame(pd.concat(
    [pd.DataFrame(X_val, columns=["a", "k h", "h", "Omega_m"]),
     pd.DataFrame(y_val, columns=["delta_m", "delta_prime_m"])], axis=1)
                             )
validation_df.to_csv(f"{path_folder}/{n}/y_val_{n}.csv", index=False)

# scaling using training set stats. We will save the scalers to apply the same transformation to the test set and future data.

# scaler_X = RobustScaler() #median and IQR
# scaler_y = RobustScaler()

scaler_X = StandardScaler() #mean and std
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)  # just transform, no fit for validation

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)  # just transform, no fit for validation

# to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 256
num_workers = 2 if torch.cuda.is_available() else 0
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers,
)

# ===========================
# 2. Defining the model


model_config = {
    'activation': 'tanh',
    'hidden_layers': (192, 192, 192, 128, 64),
    'dropout': 0.0,
    'use_layernorm': False,
}

model = ImprovedRegressionNN(**model_config).to(device)
print(f" Modelo movido a: {next(model.parameters()).device}")
#print NN architecture
print("="*50)
print(" Arquitectura de la red:")
print(model)

# ===========================
# 3. loss function and optimizer
# Targets are standardized, so MAPE is not a good training signal here.
criterion = nn.MSELoss()
LR = 6e-4 # initial LR
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # minimize val_loss
    factor=0.6,          # reduce LR by this factor
    patience=15,         # wait epochs without improvement
    min_lr=1e-7          # minimum LR
)

# Early stopping
best_val_loss = float('inf')
patience_early = 80
wait_early = 0
best_model_state = None

# ===========================
# 4. training

epochs = 800
train_losses, val_losses = [], []
lr_history = []  # to save learning rate history

for epoch in tqdm(range(epochs)):
    # training
    model.train()
    train_loss = 0
    train_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)  # Mover a GPU
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        train_samples += X_batch.size(0)
    train_loss /= train_samples

    # validation
    model.eval()
    val_loss = 0
    val_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)  # Mover a GPU
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_samples += X_batch.size(0)
    val_loss /= val_samples

    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait_early = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        wait_early += 1
        
    if wait_early >= patience_early and best_model_state is not None:
        print(f"Early stopping at epoch {epoch+1}")
        model.load_state_dict(best_model_state)
        break

    # Save history
    current_lr = optimizer.param_groups[0]['lr']
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    lr_history.append(current_lr)

    
    # GPU monitoring every 10 epochs
    if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | LR: {current_lr:.8f} | GPU: {gpu_memory:.2f}GB")


# ===========================
# 5.Save history and final metrics
# ===========================

# eval final validation metrics
model.eval()
y_true_list, y_pred_list = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        y_true_list.append(y_batch.cpu().numpy())
        y_pred_list.append(preds.cpu().numpy())

y_true = np.vstack(y_true_list)
y_pred = np.vstack(y_pred_list)

# convert to phisical values
y_true_phys = scaler_y.inverse_transform(y_true)
y_pred_phys = scaler_y.inverse_transform(y_pred)

# some metrics
mae_targets = mean_absolute_error(y_true_phys, y_pred_phys, multioutput="raw_values")
r2_targets = r2_score(y_true_phys, y_pred_phys, multioutput="raw_values")
rmse_targets = np.sqrt(mean_squared_error(y_true_phys, y_pred_phys, multioutput="raw_values"))



# ===========================
# 6. save all
# ===========================




with open(f"{path_folder}/{n}/final_metrics_{n}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Target", "MAE", "RMSE", "R2"])
    for name, mae, rmse, r2 in zip(["delta_m", "delta_prime_m"], mae_targets, rmse_targets, r2_targets):
        writer.writerow([name, mae, rmse, r2])

with open(f"{path_folder}/{n}/training_history_{n}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE", "Learning_Rate"])
    for epoch, (tr, val, lr) in enumerate(zip(train_losses, val_losses, lr_history), 1):
        writer.writerow([epoch, tr, val, lr])

with open(f"{path_folder}/{n}/info_{n}.txt", "w") as f:
    f.write("="*50 + "\n")
    f.write("CONFIGURACIÓN DEL ENTRENAMIENTO\n")
    f.write("="*50 + "\n")
    f.write(f"Device usado: {device}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write("\n")
    f.write("ARQUITECTURA DE LA RED:\n")
    f.write("-"*30 + "\n")
    f.write(str(model))
    f.write("\n")
    #write size of network (networks and layers)
    f.write("-"*30 + "\n")
    f.write(f"activation: {model_config['activation']}\n")
    f.write(f"input_size: {model.network[0].in_features}\n")
    f.write(f"output_size: {model.network[-1].out_features}\n")
    f.write(f"hidden_layers: {model_config['hidden_layers']}\n")
    f.write(f"dropout: {model_config['dropout']}\n")
    f.write(f"use_layernorm: {model_config['use_layernorm']}\n")
    f.write(f"num_epochs_total: {epochs}\n")
    f.write(f"num_epochs_trained: {len(train_losses)}\n")
    f.write(f"early_stopped: {'Yes' if len(train_losses) < epochs else 'No'}\n")
    f.write(f"number of training samples: {len(train_dataset)}\n")
    f.write(f"number of validation samples: {len(val_dataset)}\n")
    f.write(f"batch_size: {train_loader.batch_size}\n")
    f.write(f"optimizer: AdamW\n")
    f.write(f"initial_learning_rate: {LR}\n")
    f.write(f"final_learning_rate: {optimizer.param_groups[0]['lr']}\n")
    f.write(f"scheduler: ReduceLROnPlateau\n")
    f.write(f"scheduler_factor: 0.6\n")
    f.write(f"scheduler_patience: 15\n")
    f.write(f"early_stopping_patience: 80\n")
    f.write(f"loss_function: MSELoss\n")
    f.write("\n")
    f.write("RESULTADOS:\n")
    f.write("-"*30 + "\n")
    f.write(f"final_train_loss: {train_losses[-1]}\n")
    f.write(f"final_val_loss: {val_losses[-1]}\n")
    f.write(f"best_val_loss: {best_val_loss}\n")
    f.write("\n")
    f.write("DATOS:\n")
    f.write("-"*30 + "\n")
    f.write(f'grilla entrenada con:\n')
    f.write(f'{path_grilla}\n')




torch.save(model.state_dict(), f"{path_folder}/{n}/regression_model_{n}.pth")
joblib.dump(scaler_X, f"{path_folder}/{n}/scaler_X_{n}.pkl")
joblib.dump(scaler_y, f"{path_folder}/{n}/scaler_y_{n}.pkl")
with open(f"{path_folder}/{n}/model_config_{n}.json", "w") as f:
    json.dump(model_config, f, indent=2)


#print final summary

print("="*60)
print(f" TRAINING COMPLETED - Network: {n}")
print("="*60)
print(f"- Training epochs: {len(train_losses)}/{epochs}")
print(f"- Early stopping: {'Yes' if len(train_losses) < epochs else 'No'}")
print(f"- Final loss - Train: {train_losses[-1]:.6f} | Val: {val_losses[-1]:.6f}")
print(f"- Best val_loss: {best_val_loss:.6f}")
print(f"- Learning rate - Initial: {LR:.6f} | Final: {optimizer.param_groups[0]['lr']:.8f}")
print("="*60)
print("Generated archives:")
print(f"✅ Model: regression_model_{n}.pth")
print(f"✅ Scaler X: scaler_X_{n}.pkl")
print(f"✅ Scaler y: scaler_y_{n}.pkl")
print(f"✅ History: training_history_{n}.csv (with LR)")
print(f"✅ Metrics: final_metrics_{n}.csv")
print(f"✅ Complete info: info_{n}.txt")
print(' ')
print(f"ROSARIO CENTRAL EL MÁS GRANDE DEL INTERIOR.")
print("="*60)
