import pandas as pd
import os
import shutil
import random

# Загрузить список всех CO2-пригодных MOFs
alignn = pd.read_csv(os.path.expanduser(
    "~/mof-mls/ALIGNN/alignn_qmof_predictions.csv"
))
all_mofs = sorted(alignn['mof'].unique())
print(f"Всего MOFs в QMOF: {len(all_mofs)}")

# 10 лучших по ALIGNN (наш train) — исключаем их
train_mofs = [
    'qmof-23596df', 'qmof-2b7dbda', 'qmof-3c1d573', 'qmof-642a3e1',
    'qmof-78b934a', 'qmof-8bf6708', 'qmof-a694bf7', 'qmof-af2bbe9',
    'qmof-d15583a', 'qmof-f911a4f'
]

# Убрать train из кандидатов
candidates = [m for m in all_mofs if m not in train_mofs]
print(f"Кандидатов для выбора: {len(candidates)}")

# Фиксируем seed для воспроизводимости
random.seed(42)
valid_mofs = random.sample(candidates, 10)

print(f"\n=== Выбранные 10 случайных MOFs (valid) ===")
for i, mof in enumerate(sorted(valid_mofs), 1):
    print(f"  {i}. {mof}")

# Копировать CIF-файлы
cifs_source = os.path.expanduser("~/mof-mls/MOFs/qmof_co2_cifs")
cifs_dest = os.path.expanduser("~/mof-mls/MOFs/valid_10_cifs")
os.makedirs(cifs_dest, exist_ok=True)

copied = 0
for mof in valid_mofs:
    src = os.path.join(cifs_source, f"{mof}.cif")
    dst = os.path.join(cifs_dest, f"{mof}.cif")
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1

print(f"\nСкопировано {copied} CIF в {cifs_dest}/")

# Сохранить список
df = pd.DataFrame({'mof': sorted(valid_mofs)})
df.to_csv(os.path.expanduser("~/mof-mls/MOFs/valid_10_mofs.csv"), index=False)
print("Сохранено: valid_10_mofs.csv")

print("\n=== Следующий шаг ===")
print("Посчитать GCMC в RASPA для этих 10 MOFs")
print("Давления: 0.01, 0.05, 0.1, 0.5, 2.5, 5.0, 10.0 бар")
print("Температура: 298 K")