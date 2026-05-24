import os
import torch
import json
import pandas as pd
from jarvis.core.atoms import Atoms
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph
from tqdm import tqdm

# ---- Загрузка модели (один раз) ----
model_dir = "hmof_co2_absp_alignn"
config_file = os.path.join(model_dir, "config.json")
with open(config_file, "r") as f:
    config = json.load(f)

model_config = ALIGNNConfig(**config["model"])
model = ALIGNN(model_config)

checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
print("Чекпоинт:", checkpoint_files[0])
checkpoint_path = os.path.join(model_dir, checkpoint_files[0])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

# ---- Параметры ----
CIFS_DIR = os.path.expanduser("../MOFs/qmof_co2_cifs")
pressures = [0.01, 0.05, 0.1, 0.5, 2.5]

cif_files = sorted([f for f in os.listdir(CIFS_DIR) if f.endswith(".cif")])
# cif_files = cif_files[:10]  # TEST
print(f"Найдено CIF: {len(cif_files)}")

# ---- Пакетный инференс ----
rows = []
failed = []

for cif_name in tqdm(cif_files, desc="ALIGNN инференс"):
    mof_id = cif_name.replace(".cif", "")
    cif_path = os.path.join(CIFS_DIR, cif_name)

    try:
        atoms = Atoms.from_cif(cif_path)
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, max_neighbors=12)
        lat = torch.tensor(atoms.lattice_mat)

        with torch.no_grad():
            result = model([g, lg, lat]).numpy().flatten().tolist()

        for p, uptake in zip(pressures, result):
            rows.append({
                'mof': mof_id,
                'P_bar': p,
                'uptake_mol_kg': uptake
            })
    except Exception as e:
        failed.append((mof_id, str(e)))
        continue

# ---- Сохранение ----
df = pd.DataFrame(rows)
df.to_csv("alignn_qmof_predictions.csv", index=False)

print(f"\nУспешно: {df['mof'].nunique()} MOFs")
print(f"Ошибок: {len(failed)}")
if failed:
    print("Первые 5 ошибок:")
    for name, err in failed[:5]:
        print(f"  {name}: {err}")

print(f"\nСохранено: alignn_qmof_predictions.csv")
print(f"Всего предсказаний: {len(df)}")

# Пример изотермы
sample = df[df['mof'] == df['mof'].unique()[0]]
print(f"\nИзотерма для {sample['mof'].iloc[0]}:")
print(sample[['P_bar', 'uptake_mol_kg']].to_string(index=False))