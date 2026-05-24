import json
import pandas as pd
import os
import shutil
from tqdm import tqdm

with open('qmof_database/qmof.json') as f:
    data = json.load(f)

print(f"Всего MOFs в QMOF: {len(data)}")

rows = []
for entry in data:
    info = entry['info']
    rows.append({
        'qmof_id': entry['qmof_id'],
        'name': entry['name'],
        'formula': info.get('formula'),
        'natoms': info.get('natoms'),
        'pld': info.get('pld'),
        'lcd': info.get('lcd'),
        'density': info.get('density'),
        'volume': info.get('volume'),
        'source': info.get('source'),
        'synthesized': info.get('synthesized'),
    })

df = pd.DataFrame(rows)
print(f"\nСтатистика PLD:")
print(df['pld'].describe())

# Фильтр для CO2: PLD > 3.3 Å (кинетический диаметр)
PLD_THRESHOLD = 3.3
co2_capable = df[df['pld'] > PLD_THRESHOLD].copy()
print(f"\nMOFs с PLD > {PLD_THRESHOLD} Å: {len(co2_capable)}")

df.to_csv('qmof_all_geometry.csv', index=False)
co2_capable.to_csv('qmof_co2_capable.csv', index=False)
print("\nСохранено: qmof_all_geometry.csv, qmof_co2_capable.csv")

# Копируем CIF-файлы отфильтрованных MOFs
CIFS_SOURCE = 'qmof_database/relaxed_structures'
CIFS_DEST = 'qmof_co2_cifs'

if os.path.exists(CIFS_SOURCE):
    os.makedirs(CIFS_DEST, exist_ok=True)
    print(f"\nКопирую CIF-файлы в {CIFS_DEST}/ ...")
    copied = 0
    for qmof_id in tqdm(co2_capable['qmof_id']):
        src = f"{CIFS_SOURCE}/{qmof_id}.cif"
        dst = f"{CIFS_DEST}/{qmof_id}.cif"
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
    print(f"Скопировано {copied} CIF-файлов")
else:
    print(f"\nПапка {CIFS_SOURCE} не найдена.")
    print("Нужно распаковать relaxed_structures.zip:")
    print("  cd qmof_database && unzip relaxed_structures.zip")