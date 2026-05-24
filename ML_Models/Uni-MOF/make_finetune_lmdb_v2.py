import os, re, pickle, lmdb, numpy as np
from pathlib import Path
from pymatgen.core import Structure
import pandas as pd

# Пути — для запуска внутри Docker
TRAIN_CIFS = Path("/workspace/MOFs/selected_10_cifs")
VALID_CIFS = Path("/workspace/MOFs/valid_10_cifs")
TRAIN_CSV = "/workspace/MOFs/NEW_RASPA_selected_10_cifs.csv"
VALID_CSV = "/workspace/MOFs/NEW_RASPA_valid_10_cifs.csv"
OUT_TRAIN = "/workspace/Uni-MOF/data/finetune_v2/hmof/train.lmdb"
OUT_VALID = "/workspace/Uni-MOF/data/finetune_v2/hmof/valid.lmdb"
DICT_SRC  = "/workspace/Uni-MOF/data/dict.txt"
DICT_DST  = "/workspace/Uni-MOF/data/finetune_v2/hmof/dict.txt"

GAS = "CO2"
GAS_ID = {"CH4":1, "CO2":2, "Ar":3, "Kr":4, "Xe":5, "O2":6, "N2":7}
GAS_ATTR = {"CO2": [304.13, 73.77, 0.2239, 44.01, 216.58, 194.70]}

COMMON_P = [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]

def norm(a):
    return re.sub(r"\d+", "", a)

def parse(cif_path):
    s = Structure.from_file(str(cif_path), primitive=False)
    df = s.as_dataframe()
    return dict(
        ID=cif_path.stem,
        atoms=df["Species"].astype(str).map(norm).tolist(),
        coordinates=df[["x","y","z"]].values.astype(np.float32),
        abc=s.lattice.abc,
        angles=s.lattice.angles,
        volume=s.lattice.volume,
        lattice_matrix=s.lattice.matrix,
        abc_coordinates=df[["a","b","c"]].values.astype(np.float32),
    )

def write_lmdb(gcmc_csv, cifs_dir, out_path, common_p):
    gcmc = pd.read_csv(gcmc_csv)
    # Фильтровать только общие давления
    gcmc = gcmc[gcmc['P'].isin(common_p)]
    mofs = sorted(gcmc['MOF'].unique())
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    
    env = lmdb.open(out_path, subdir=False, readonly=False, lock=False,
                    readahead=False, meminit=False, map_size=int(1e10))
    txn = env.begin(write=True)
    
    idx = 0
    success = 0
    for mof_id in mofs:
        cif_path = cifs_dir / f"{mof_id}.cif"
        if not cif_path.exists():
            print(f"  НЕ НАЙДЕН: {mof_id}")
            continue
        
        try:
            base = parse(cif_path)
        except Exception as e:
            print(f"  ОШИБКА {mof_id}: {e}")
            continue
        
        mof_gcmc = gcmc[gcmc['MOF'] == mof_id]
        for _, row in mof_gcmc.iterrows():
            P_pa = row['P'] * 1e5
            target_cm3g = row['Abs_mol_kg'] * 22.414  # mol/kg -> cm3(STP)/g
            
            d = dict(base)
            d["gas"]         = np.array(GAS_ID[GAS], dtype=np.int32)
            d["gas_attr"]    = np.array(GAS_ATTR[GAS], dtype=np.float32)
            d["temperature"] = np.array(298.0, dtype=np.float32)
            d["pressure"]    = np.array(np.log10(P_pa), dtype=np.float32)
            d["target"]      = np.array(target_cm3g, dtype=np.float32)
            d["task_name"]   = f"{mof_id}#{GAS}#298.0#{P_pa}"
            
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(d, protocol=-1))
            idx += 1
        success += 1
    
    txn.commit()
    env.close()
    return success, idx

# Создать LMDB
print("=== Train (10 лучших MOFs) ===")
n_mofs, n_points = write_lmdb(TRAIN_CSV, TRAIN_CIFS, OUT_TRAIN, COMMON_P)
print(f"MOFs: {n_mofs}, точек: {n_points}")

print("\n=== Valid (10 случайных MOFs) ===")
n_mofs, n_points = write_lmdb(VALID_CSV, VALID_CIFS, OUT_VALID, COMMON_P)
print(f"MOFs: {n_mofs}, точек: {n_points}")

# Скопировать dict.txt
import shutil
os.makedirs(os.path.dirname(DICT_DST), exist_ok=True)
shutil.copy(DICT_SRC, DICT_DST)
print(f"\ndict.txt скопирован")
print("Готово!")