import pickle
import pandas as pd
import torch

with open('/workspace/Uni-MOF/дообучение/infer_out_finetuned/qmof_finetune_test.out.pkl', 'rb') as f:
    data = pickle.load(f)

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

print(f"Всего: {len(df)} предсказаний, {df['mof'].nunique()} MOFs")
print(f"Диапазон: {df['uptake_cm3g'].min():.2f} - {df['uptake_cm3g'].max():.2f} cm3/g")
print(f"Диапазон: {df['uptake_mol_kg'].min():.4f} - {df['uptake_mol_kg'].max():.4f} mol/kg")

# Изотерма для первого MOF
sample = df[df['mof'] == df['mof'].unique()[0]].sort_values('P_bar')
print(f"\nИзотерма для {sample['mof'].iloc[0]}:")
print(sample[['P_bar', 'uptake_mol_kg']].to_string(index=False))

df.to_csv('unimof_qmof_predictions_finetuned.csv', index=False)
print(f"\nСохранено: unimof_qmof_predictions_finetuned.csv")