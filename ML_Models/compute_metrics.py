"""
Расчёт метрик качества моделей ALIGNN и Uni-MOF
относительно GCMC (ground truth) ДО дообучения.

Метрики:
  MAE  = (1/N) * Σ|ŷᵢ - yᵢ|         — средняя абсолютная ошибка
  MSE  = (1/N) * Σ(ŷᵢ - yᵢ)²        — среднеквадратичная ошибка
  RMSE = √MSE = √((1/N) * Σ(ŷᵢ - yᵢ)²) — корень из MSE

где yᵢ — значение GCMC, ŷᵢ — предсказание модели, N — число точек.
"""

import pandas as pd
import numpy as np


def compute_metrics(predicted, actual):
    """
    Считает MAE, MSE, RMSE между предсказаниями и ground truth.

    Параметры:
        predicted: pd.Series — предсказанные значения (ŷ)
        actual:    pd.Series — истинные значения GCMC (y)

    Возвращает:
        dict с ключами MAE, MSE, RMSE
    """
    diff = predicted - actual                     # ŷᵢ - yᵢ
    mae  = diff.abs().mean()                      # (1/N) * Σ|ŷᵢ - yᵢ|
    mse  = (diff ** 2).mean()                     # (1/N) * Σ(ŷᵢ - yᵢ)²
    rmse = np.sqrt(mse)                           # √MSE
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


# ============================================================
# Загрузка данных
# ============================================================
gcmc   = pd.read_csv("/home/chernysheva/mof-mls/MOFs/RASPA_results_10mof(1).csv")
alignn = pd.read_csv("/home/chernysheva/mof-mls/ALIGNN/alignn_qmof_predictions.csv")
unimof = pd.read_csv("/home/chernysheva/mof-mls/Uni-MOF/unimof_qmof_predictions_finetuned.csv")

# Обнулить отрицательные предсказания ALIGNN (нефизичные артефакты)
alignn.loc[alignn["uptake_mol_kg"] < 0, "uptake_mol_kg"] = 0.0

# 10 выбранных MOFs и общие давления
mofs_10 = gcmc["MOF"].unique()
common_p = [0.01, 0.05, 0.1, 0.5, 2.5]

# Переименовать колонки GCMC для удобства объединения
gcmc_renamed = gcmc.rename(columns={
    "MOF": "mof", "P": "P_bar", "Abs_mol_kg": "gcmc_mol_kg"
})

# ============================================================
# ALIGNN vs GCMC
# ============================================================
alignn_10 = alignn[
    alignn["mof"].isin(mofs_10) & alignn["P_bar"].isin(common_p)
]
merged_a = alignn_10.merge(
    gcmc_renamed[["mof", "P_bar", "gcmc_mol_kg"]], on=["mof", "P_bar"]
)
metrics_a = compute_metrics(merged_a["uptake_mol_kg"], merged_a["gcmc_mol_kg"])

# ============================================================
# Uni-MOF vs GCMC
# ============================================================
unimof_10 = unimof[
    unimof["mof"].isin(mofs_10) & unimof["P_bar"].round(2).isin(common_p)
]
merged_u = unimof_10.merge(
    gcmc_renamed[["mof", "P_bar", "gcmc_mol_kg"]], on=["mof", "P_bar"]
)
metrics_u = compute_metrics(merged_u["uptake_mol_kg"], merged_u["gcmc_mol_kg"])

# ============================================================
# Вывод результатов
# ============================================================
print("=== Метрики до дообучения ===\n")
print(f"{'Метрика':<8} {'ALIGNN':>10} {'Uni-MOF':>10}")
print("-" * 30)
for key in ["MAE", "MSE", "RMSE"]:
    print(f"{key:<8} {metrics_a[key]:>10.4f} {metrics_u[key]:>10.4f}")

# MAE по каждому MOF + диапазон GCMC для интерпретации
print(f"\n{'MOF':<14} {'ALIGNN MAE':>13} {'Uni-MOF MAE':>13}")
print("-" * 45)
for mof in sorted(mofs_10):
    ma = merged_a[merged_a["mof"] == mof]
    mu = merged_u[merged_u["mof"] == mof]
    mae_a = compute_metrics(ma["uptake_mol_kg"], ma["gcmc_mol_kg"])["MAE"]
    mae_u = compute_metrics(mu["uptake_mol_kg"], mu["gcmc_mol_kg"])["MAE"]
    print(f"{mof:<14} {mae_a:>12.4f} {mae_u:>12.4f}")

# Сохранить
result = pd.DataFrame({
    "Модель": ["ALIGNN", "Uni-MOF"],
    "MAE":  [metrics_a["MAE"],  metrics_u["MAE"]],
    "MSE":  [metrics_a["MSE"],  metrics_u["MSE"]],
    "RMSE": [metrics_a["RMSE"], metrics_u["RMSE"]],
})
result.to_csv("metrics_after_finetune.csv", index=False)
print("\nСохранено: metrics_after_finetune.csv")