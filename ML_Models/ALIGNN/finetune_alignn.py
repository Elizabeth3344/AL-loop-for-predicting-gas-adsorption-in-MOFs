import os
import json
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Датасет
# ============================================================
class MOFDataset(Dataset):
    def __init__(self, csv_path, cifs_dir):
        self.df = pd.read_csv(csv_path)
        self.cifs_dir = cifs_dir
        self.target_cols = [c for c in self.df.columns if c.startswith('target_')]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mof_id = row['id']
        targets = torch.tensor([row[c] for c in self.target_cols], dtype=torch.float32)
        
        cif_path = os.path.join(self.cifs_dir, f"{mof_id}.cif")
        atoms = Atoms.from_cif(cif_path)
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, max_neighbors=12)
        lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        
        return g, lg, lat, targets, mof_id

def collate_fn(batch):
    # Каждый MOF обрабатывается отдельно (batch_size=1 для графов разного размера)
    return batch[0]

# ============================================================
# Загрузка модели
# ============================================================
model_dir = "../hmof_co2_absp_alignn"
config_file = os.path.join(model_dir, "config.json")
with open(config_file, "r") as f:
    config = json.load(f)

model_config = ALIGNNConfig(**config["model"])
model = ALIGNN(model_config)

checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
checkpoint_path = os.path.join(model_dir, checkpoint_files[0])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
print(f"Загружена модель из {checkpoint_path}")

# ============================================================
# Параметры дообучения
# ============================================================
FINETUNE_DIR = "finetune_data"
LR = 1e-5
EPOCHS = 50
SAVE_DIR = "finetune_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Данные
# ============================================================
train_dataset = MOFDataset(
    os.path.join(FINETUNE_DIR, "train.csv"),
    os.path.join(FINETUNE_DIR, "cifs")
)
val_dataset = MOFDataset(
    os.path.join(FINETUNE_DIR, "val.csv"),
    os.path.join(FINETUNE_DIR, "cifs")
)

print(f"Train: {len(train_dataset)} MOFs")
print(f"Valid: {len(val_dataset)} MOFs")

# ============================================================
# Обучение
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

model.train()
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_losses = []
    for i in range(len(train_dataset)):
        g, lg, lat, targets, mof_id = train_dataset[i]
        
        optimizer.zero_grad()
        pred = model([g, lg, lat])
        pred = pred.squeeze()
        
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    avg_train_loss = np.mean(train_losses)
    
    # --- Validation ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            g, lg, lat, targets, mof_id = val_dataset[i]
            pred = model([g, lg, lat]).squeeze()
            loss = loss_fn(pred, targets)
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    
    # Сохранить лучшую модель по валидации
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'val_loss': best_val_loss
        }, os.path.join(SAVE_DIR, "best_model.pt"))
        marker = " *BEST*"
    else:
        marker = ""
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f}{marker}")

# Сохранить финальную модель
torch.save({
    'model': model.state_dict(),
    'epoch': EPOCHS,
}, os.path.join(SAVE_DIR, "last_model.pt"))

print(f"\nОбучение завершено!")
print(f"Лучший val_loss: {best_val_loss:.4f}")
print(f"Чекпоинты в {SAVE_DIR}/")