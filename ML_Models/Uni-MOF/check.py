import pickle
import torch
import pandas as pd

with open('/workspace/Uni-MOF/infer_out/weights_test.out.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Батчей: {len(data)}")

rows = []
for batch in data:
    predictions = batch['predict'].cpu().float().numpy().flatten()
    task_names = batch['task_name']
    
    for task_name, value in zip(task_names, predictions):
        parts = task_name.split('#')
        mof_id = parts[0]
        pressure_pa = float(parts[-1])
        rows.append({
            'mof': mof_id,
            'P_Pa': pressure_pa,
            'P_bar': pressure_pa / 1e5,
            'uptake_cm3g': float(value)
        })

df = pd.DataFrame(rows)
df['uptake_mol_kg'] = df['uptake_cm3g'] * 1000 / 22414

print(f"Всего предсказаний: {len(df)}")
print(f"MOFs: {df['mof'].nunique()}")
print(f"Давлений: {df['P_bar'].nunique()}")
print(f"Диапазон: {df['uptake_cm3g'].min():.2f} - {df['uptake_cm3g'].max():.2f} cm3/g")

# Изотерма для первого MOF
sample_mof = df['mof'].unique()[0]
one_iso = df[df['mof'] == sample_mof].sort_values('P_bar')
print(f"\nИзотерма для {sample_mof}:")
print(one_iso[['P_bar', 'uptake_cm3g', 'uptake_mol_kg']].to_string(index=False))

df.to_csv('unimof_qmof_predictions.csv', index=False)
print(f"\nСохранено: unimof_qmof_predictions.csv")