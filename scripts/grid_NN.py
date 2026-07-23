import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import csv
from tqdm import tqdm
import sys
sys.path.append('/home/pedrorozin/paper_tesis2025/scripts')
from funciones_tesis import ImprovedRegressionNN
import os

# ===========================
# GPU Configuration
# ===========================
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
n = 'NN_z_approx_32_trained_big_grid'

if os.path.exists(path_folder + n):
    raise FileExistsError(f"El directorio {path_folder}/{n} ya existe.")

if not os.path.exists(path_folder + n):
    os.makedirs(path_folder + n)


# ===========================
# 1. load data y split and scale
# ===========================

#select grid for training
main_path = '/home/pedrorozin/paper_tesis2025/outputs/grids/'
name_grid = 'grid_z_approx_32_training_data_big'
path_grilla = f'{main_path}{name_grid}/grilla_results_{name_grid}.csv'
df_grilla = pd.read_csv(path_grilla)
mask = (df_grilla['k h'] < 0.21) & (df_grilla['a'] < 0.035) #this is important for numerical reasons. its crucial that the grid has a dense ammount of values in the desire regions
df = df_grilla[mask].copy()

# Features y targets

#filter features with k h <= 0.25 (no lineal regime)
features = df[["a", "k h", "h", "Omega_m"]][df['k h'] <= 0.4].values

targets = df[["delta_m", "delta_prime_m"]].values

# Split train y val (80% train, 20% val) before scaling.

X_train, X_val, y_train, y_val = train_test_split(
    features, targets, 
    test_size=0.2, 
    random_state=100, 
    shuffle=True
)

#save y_val to csv for later evaluation
pd.DataFrame(y_val, columns=["delta_m", "delta_prime_m"]).to_csv(f"{path_folder}/{n}/y_val_{n}.csv", index=False)

# scaling using training set stats. We will save the scalers to apply the same transformation to the test set and future data.

scaler_X = RobustScaler() #median and IQR
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)  # just transform, no fit for validation

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)  # just transform, no fit for validation

# to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

# datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# ===========================
# 2. Defining the model
# ===========================

model = ImprovedRegressionNN(activation='tanh').to(device)
print(f" Modelo movido a: {next(model.parameters()).device}")

# ===========================
# 3. loss function and optimizer
# ===========================
criterion = nn.MSELoss()
LR = 1e-3 #initial LR
optimizer = optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # minimize val_loss
    factor=0.6,          # reduce LR by this factor
    patience=15,         # wait 15 epochs without improvement
    min_lr=1e-7          # minimum LR
)

# Early stopping
best_val_loss = float('inf')
patience_early = 50
wait_early = 0
best_model_state = None

# ===========================
# 4. training
# ===========================
epochs = 800
train_losses, val_losses = [], []
lr_history = []  # to save learning rate history

for epoch in range(epochs):
    # training
    model.train()
    train_loss = 0
    train_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Mover a GPU
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Mover a GPU
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
        best_model_state = model.state_dict().copy()
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
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.8f} | GPU: {gpu_memory:.2f}GB")
    else:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.8f}")

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
    f.write(f"activation: {model.network[1]}\n")  # si no son todas iguales, cambiar
    f.write(f"input_size: {model.network[0].in_features}\n")
    f.write(f"output_size: {model.network[-1].out_features}\n")
    f.write(f"num_epochs_total: {epochs}\n")
    f.write(f"num_epochs_trained: {len(train_losses)}\n")
    f.write(f"early_stopped: {'Yes' if len(train_losses) < epochs else 'No'}\n")
    f.write(f"number of training samples: {len(train_dataset)}\n")
    f.write(f"number of validation samples: {len(val_dataset)}\n")
    f.write(f"batch_size: {train_loader.batch_size}\n")
    f.write(f"optimizer: Adam\n")
    f.write(f"initial_learning_rate: {LR}\n")
    f.write(f"final_learning_rate: {optimizer.param_groups[0]['lr']}\n")
    f.write(f"scheduler: ReduceLROnPlateau\n")
    f.write(f"scheduler_factor: 0.6\n")
    f.write(f"scheduler_patience: 20\n")
    f.write(f"early_stopping_patience: 50\n")
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
