import pickle
import pandas as pd
import numpy as np
import torch
import os

# ============================================================
# 1. Распарсить предсказания
# ============================================================
pkl_path = '/workspace/Uni-MOF/дообучение/infer_out_v2/qmof_finetune_v2_test.out.pkl'
with open(pkl_path, 'rb') as f:
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
            'P_bar': pressure_pa / 1e5,
            'uptake_mol_kg': float(value) * 1000 / 22414
        })

after = pd.DataFrame(rows)
print(f"Всего предсказаний: {len(after)}, MOFs: {after['mof'].nunique()}")

# ============================================================
# 2. Загрузить остальные данные
# ============================================================
gcmc_valid = pd.read_csv('/workspace/MOFs/NEW_RASPA_valid_10_cifs.csv')
before = pd.read_csv('/workspace/Uni-MOF/unimof_qmof_predictions.csv')

gcmc_r = gcmc_valid.rename(columns={'MOF': 'mof', 'P': 'P_bar', 'Abs_mol_kg': 'gcmc'})
valid_mofs = gcmc_r['mof'].unique()
common_p = [0.01, 0.05, 0.1, 0.5, 2.5, 5.0, 10.0]

gcmc_common = gcmc_r[gcmc_r['P_bar'].isin(common_p)]

def compute_metrics(predicted, actual):
    diff = predicted - actual
    mae  = diff.abs().mean()
    mse  = (diff ** 2).mean()
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}

# ============================================================
# 3. Метрики на valid MOFs — все давления
# ============================================================
merged_b = before[before['mof'].isin(valid_mofs) & before['P_bar'].round(2).isin(common_p)].merge(
    gcmc_common[['mof', 'P_bar', 'gcmc']], on=['mof', 'P_bar'])
merged_a = after[after['mof'].isin(valid_mofs) & after['P_bar'].round(2).isin(common_p)].merge(
    gcmc_common[['mof', 'P_bar', 'gcmc']], on=['mof', 'P_bar'])

metrics_b = compute_metrics(merged_b['uptake_mol_kg'], merged_b['gcmc'])
metrics_a = compute_metrics(merged_a['uptake_mol_kg'], merged_a['gcmc'])

print(f"\n=== Uni-MOF vs GCMC — Valid (10 MOFs), все давления ===")
print(f"{'Метрика':<8} {'До':>12} {'После':>12} {'Изменение':>12}")
print("-" * 46)
for key in ["MAE", "MSE", "RMSE"]:
    change = ((metrics_a[key] - metrics_b[key]) / metrics_b[key]) * 100
    print(f"{key:<8} {metrics_b[key]:>12.4f} {metrics_a[key]:>12.4f} {change:>+11.1f}%")

# ============================================================
# 4. Метрики на valid MOFs — только 0.1 бар
# ============================================================
merged_b_01 = merged_b[merged_b['P_bar'].round(2) == 0.1]
merged_a_01 = merged_a[merged_a['P_bar'].round(2) == 0.1]

metrics_b_01 = compute_metrics(merged_b_01['uptake_mol_kg'], merged_b_01['gcmc'])
metrics_a_01 = compute_metrics(merged_a_01['uptake_mol_kg'], merged_a_01['gcmc'])

print(f"\n=== Uni-MOF vs GCMC — Valid (10 MOFs), только 0.1 бар ===")
print(f"{'Метрика':<8} {'До':>12} {'После':>12} {'Изменение':>12}")
print("-" * 46)
for key in ["MAE", "MSE", "RMSE"]:
    change = ((metrics_a_01[key] - metrics_b_01[key]) / metrics_b_01[key]) * 100
    print(f"{key:<8} {metrics_b_01[key]:>12.4f} {metrics_a_01[key]:>12.4f} {change:>+11.1f}%")

# ============================================================
# 5. Детально: изотермы valid MOFs
# ============================================================
print(f"\n=== Изотермы valid MOFs ===")
for mof in sorted(valid_mofs)[:3]:
    print(f"\n{mof}:")
    print(f"  {'P(bar)':>7} {'GCMC':>10} {'До':>10} {'После':>10}")
    for p in common_p:
        g = gcmc_common[(gcmc_common['mof'] == mof) & (gcmc_common['P_bar'] == p)]['gcmc']
        b = merged_b[(merged_b['mof'] == mof) & (merged_b['P_bar'].round(2) == p)]['uptake_mol_kg']
        a = merged_a[(merged_a['mof'] == mof) & (merged_a['P_bar'].round(2) == p)]['uptake_mol_kg']
        g_val = g.values[0] if len(g) > 0 else float('nan')
        b_val = b.values[0] if len(b) > 0 else float('nan')
        a_val = a.values[0] if len(a) > 0 else float('nan')
        print(f"  {p:>7.2f} {g_val:>10.4f} {b_val:>10.4f} {a_val:>10.4f}")

after.to_csv('/workspace/Uni-MOF/unimof_qmof_predictions_finetuned_v2.csv', index=False)
print(f"\nСохранено: unimof_qmof_predictions_finetuned_v2.csv")

