import os
import pandas as pd
import numpy as np
import shutil

# Загрузка
alignn = pd.read_csv(
    "/home/chernysheva/mof-mls/ALIGNN/alignn_qmof_predictions.csv"
)
unimof = pd.read_csv(
    "/home/chernysheva/mof-mls/Uni-MOF/unimof_qmof_predictions.csv"
)

# Обнулить отрицательную адсорбцию
alignn.loc[alignn['uptake_mol_kg'] < 0, 'uptake_mol_kg'] = 0.0

# ALIGNN при 2.5 бар — рейтинг MOFs
alignn_25 = alignn[alignn['P_bar'] == 2.5].copy()
alignn_25 = alignn_25.sort_values('uptake_mol_kg', ascending=False)

print(f"=== Топ-20 MOFs по uptake ALIGNN при 2.5 бар ===")
print(alignn_25.head(20)[['mof', 'uptake_mol_kg']].to_string(index=False))

# Выбираем топ-10
top10_mofs = alignn_25.head(10)['mof'].tolist()

# Показать изотермы обеих моделей для выбранных MOFs
# Uni-MOF: оставить общие давления
common_p = [0.01, 0.05, 0.1, 0.5, 2.5]
unimof_common = unimof[unimof['P_bar'].round(2).isin(common_p)]

print(f"\n=== Выбранные 10 MOFs: сравнение ALIGNN vs Uni-MOF ===")
for mof in top10_mofs:
    iso_a = alignn[alignn['mof'] == mof].sort_values('P_bar')
    iso_u = unimof_common[unimof_common['mof'] == mof].sort_values('P_bar')

    print(f"\n{mof}:")
    print(f"  P(bar)  ALIGNN(mol/kg)  Uni-MOF(mol/kg)")
    for _, row_a in iso_a.iterrows():
        p = row_a['P_bar']
        a_val = row_a['uptake_mol_kg']
        u_row = iso_u[iso_u['P_bar'].round(2) == round(p, 2)]
        u_val = u_row['uptake_mol_kg'].values[0] if len(u_row) > 0 else float('nan')
        print(f"  {p:5.2f}   {a_val:12.3f}   {u_val:12.3f}")

# Сохранить
result = pd.DataFrame({'mof': top10_mofs})
result = result.merge(
    alignn_25[['mof', 'uptake_mol_kg']],
    on='mof'
)
result.columns = ['mof', 'alignn_uptake_2.5bar']
result.to_csv("selected_10_mofs.csv", index=False)
print(f"\nСохранено: selected_10_mofs.csv")

# #Закидываем cif выбранных моф в отдельную папку для расчета в RASPA
# cifs_source = os.path.expanduser("~/mof-mls/MOFs/qmof_co2_cifs")
# cifs_dest = os.path.expanduser("~/mof-mls/MOFs/selected_10_cifs")
# os.makedirs(cifs_dest, exist_ok=True)

# copied = 0
# for mof_id in top10_mofs:
#     src = os.path.join(cifs_source, f"{mof_id}.cif")
#     dst = os.path.join(cifs_dest, f"{mof_id}.cif")
#     if os.path.exists(src):
#         shutil.copy(src, dst)
#         copied += 1
#         print(f"  Скопирован: {mof_id}.cif")
#     else:
#         print(f"  НЕ НАЙДЕН: {mof_id}.cif")

# print(f"\nСкопировано {copied} CIF в {cifs_dest}/")