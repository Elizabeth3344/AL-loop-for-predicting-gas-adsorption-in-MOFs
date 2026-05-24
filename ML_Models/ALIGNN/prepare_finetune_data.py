import pandas as pd
import os
import shutil

GCMC_CSV = os.path.expanduser("~/mof-mls/MOFs/NEW_RASPA_selected_10_cifs.csv")
CIFS_SOURCE = os.path.expanduser("~/mof-mls/MOFs/selected_10_cifs")
OUTPUT_DIR = os.path.expanduser("~/mof-mls/ALIGNN/дообучение/finetune_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cifs"), exist_ok=True)

gcmc = pd.read_csv(GCMC_CSV)
pressures = [0.01, 0.05, 0.1, 0.5, 2.5]

# Для каждого MOF собрать 5 значений адсорбции
rows = []
for mof in sorted(gcmc['MOF'].unique()):
    mof_data = gcmc[gcmc['MOF'] == mof]
    values = {}
    for p in pressures:
        match = mof_data[mof_data['P'] == p]
        if len(match) > 0:
            values[p] = match['Abs_mol_kg'].values[0]
    
    if len(values) == 5:
        rows.append({
            'id': mof,
            'target_0.01': values[0.01],
            'target_0.05': values[0.05],
            'target_0.1': values[0.1],
            'target_0.5': values[0.5],
            'target_2.5': values[2.5],
        })
        # Копировать CIF
        src = os.path.join(CIFS_SOURCE, f"{mof}.cif")
        dst = os.path.join(OUTPUT_DIR, "cifs", f"{mof}.cif")
        if os.path.exists(src):
            shutil.copy(src, dst)

df = pd.DataFrame(rows)
print(f"Train MOFs с полными данными: {len(df)}")
print(df.to_string(index=False))

train_df = df

# ============================================================
# Valid — 10 рандомных MOFs
# ============================================================
VALID_CSV = os.path.expanduser("~/mof-mls/MOFs/NEW_RASPA_valid_10_cifs.csv")
VALID_CIFS = os.path.expanduser("~/mof-mls/MOFs/valid_10_cifs")

valid_gcmc = pd.read_csv(VALID_CSV)
valid_rows = []
for mof in sorted(valid_gcmc['MOF'].unique()):
    mof_data = valid_gcmc[valid_gcmc['MOF'] == mof]
    values = {}
    for p in pressures:
        match = mof_data[mof_data['P'] == p]
        if len(match) > 0:
            values[p] = match['Abs_mol_kg'].values[0]
    
    if len(values) == 5:
        valid_rows.append({
            'id': mof,
            'target_0.01': values[0.01],
            'target_0.05': values[0.05],
            'target_0.1': values[0.1],
            'target_0.5': values[0.5],
            'target_2.5': values[2.5],
        })
        # Копировать CIF valid MOFs в ту же папку cifs
        src = os.path.join(VALID_CIFS, f"{mof}.cif")
        dst = os.path.join(OUTPUT_DIR, "cifs", f"{mof}.cif")
        if os.path.exists(src):
            shutil.copy(src, dst)

val_df = pd.DataFrame(valid_rows)
print(f"\nValid MOFs с полными данными: {len(val_df)}")
print(val_df.to_string(index=False))

print(f"\nTrain: {len(train_df)} MOFs")
print(f"Valid: {len(val_df)} MOFs")

# Сохранить
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)

print(f"\nСохранено в {OUTPUT_DIR}/")