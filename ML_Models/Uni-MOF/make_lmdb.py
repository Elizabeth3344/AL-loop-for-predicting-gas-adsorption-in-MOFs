import os, re, pickle, lmdb, numpy as np
from pathlib import Path
from pymatgen.core import Structure
from tqdm import tqdm

CIFS_DIR = Path("/workspace/qmof_co2_cifs")  # уточнить путь на сервере
OUT      = "/workspace/data/qmof/qmof_co2_test.lmdb"
GAS      = "CO2"
T_K      = 298.0

# Давления: 6 общих с ALIGNN + 2 дополнительных до 10 бар
PRESS_BAR = [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
PRESS_PA  = np.array(PRESS_BAR) * 1e5

GAS_ID   = {"CH4":1, "CO2":2, "Ar":3, "Kr":4, "Xe":5, "O2":6, "N2":7}
GAS_ATTR = {"CO2": [304.13, 73.77, 0.2239, 44.01, 216.58, 194.70]}

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

cif_files = sorted(CIFS_DIR.glob("*.cif"))
# cif_files = cif_files[:10]  # TEST: только первые 10
print(f"Найдено CIF-файлов: {len(cif_files)}")
print(f"Давлений на каждый MOF: {len(PRESS_PA)}")
print(f"Всего записей будет: {len(cif_files) * len(PRESS_PA)}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT):
    os.remove(OUT)

env = lmdb.open(OUT, subdir=False, readonly=False, lock=False,
                readahead=False, meminit=False, map_size=int(1e11))
txn = env.begin(write=True)

success = 0
failed = []
entry_idx = 0

for mof_idx, cif in enumerate(tqdm(cif_files, desc="Обработка CIF")):
    try:
        base = parse(cif)
    except Exception as e:
        failed.append((cif.name, str(e)))
        continue

    for P in PRESS_PA:
        d = dict(base)
        d["gas"]         = np.array(GAS_ID[GAS], dtype=np.int32)
        d["gas_attr"]    = np.array(GAS_ATTR[GAS], dtype=np.float32)
        d["temperature"] = np.array(T_K, dtype=np.float32)
        d["pressure"]    = np.array(np.log10(P), dtype=np.float32)
        d["target"]      = np.array(0.0, dtype=np.float32)
        d["task_name"]   = f"{d['ID']}#{GAS}#{T_K}#{P}"

        txn.put(f"{entry_idx}".encode("ascii"), pickle.dumps(d, protocol=-1))
        entry_idx += 1

    success += 1

    # Периодически коммитим, чтобы не потерять данные при сбое
    if success % 500 == 0:
        txn.commit()
        txn = env.begin(write=True)

txn.commit()
env.close()

print(f"\nУспешно обработано MOFs: {success}")
print(f"Ошибок: {len(failed)}")
if failed:
    print(f"Первые 5 ошибок:")
    for name, err in failed[:5]:
        print(f"  {name}: {err}")

print(f"\nLMDB: {OUT}")
print(f"Всего записей: {entry_idx}")