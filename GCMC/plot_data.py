import pandas as pd
import matplotlib.pyplot as plt
import sys

COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
LOW_P = 0.01  # выделяемое давление на графике Q_st


def _isotherm(data, mof, col, ylabel, title_suffix, filename_suffix):
    mof_data = data[data["MOF"] == mof]
    temps = sorted(mof_data["T"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, temp in enumerate(temps):
        subset = mof_data[mof_data["T"] == temp].sort_values("P")
        ax.plot(
            subset["P"], subset[col],
            marker='o', linestyle='-', linewidth=2, markersize=5,
            color=COLORS[i % len(COLORS)], label=f"{temp} K"
        )

    ax.set_title(f"{title_suffix}  —  {mof}", fontsize=13)
    ax.set_xlabel("Давление (бар)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(title="Температура", fontsize=9)
    plt.tight_layout()

    out = f"{filename_suffix}_{mof}.png"
    plt.savefig(out, dpi=200)
    print(f"  Сохранён: {out}")
    plt.close()


def plot_isotherms_abs(data, mof):
    _isotherm(
        data, mof,
        col="Abs_mol_kg",
        ylabel="Абсолютная адсорбция $n_{abs}$ (моль/кг)",
        title_suffix="Изотермы абсолютной адсорбции",
        filename_suffix="isotherms_abs"
    )


def plot_isotherms_exc(data, mof):
    _isotherm(
        data, mof,
        col="Exc_mol_kg",
        ylabel="Избыточная адсорбция $n_{exc}$ (моль/кг)",
        title_suffix="Изотермы избыточной адсорбции",
        filename_suffix="isotherms_exc"
    )


def plot_qst(data, mof):
    """Теплота адсорбции в зависимости от заполнения МОК"""
    mof_data = data[data["MOF"] == mof]
    temps = sorted(mof_data["T"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, temp in enumerate(temps):
        color = COLORS[i % len(COLORS)]
        subset = mof_data[mof_data["T"] == temp].sort_values("P")

        low = subset[subset["P"] == LOW_P]
        rest = subset[subset["P"] != LOW_P]

        ax.plot(
            subset["Abs_mol_kg"], subset["Qst_kJmol"],
            linestyle='-', linewidth=1.5, color=color, zorder=2
        )
        ax.scatter(
            rest["Abs_mol_kg"], rest["Qst_kJmol"],
            marker='o', s=40, color=color, zorder=3,
            label=f"{temp} K"
        )
        if not low.empty:
            ax.scatter(
                low["Abs_mol_kg"], low["Qst_kJmol"],
                marker='*', s=220, color=color, edgecolors='black',
                linewidths=0.5, zorder=5,
                label=f"{temp} K  (P={LOW_P} бар)"
            )

    ax.set_title(
        f"Теплота адсорбции в зависимости от заполнения МОК", fontsize=13)
    ax.set_xlabel("Абсолютная адсорбция $n_{abs}$ (моль/кг)", fontsize=11)
    ax.set_ylabel(r"$Q_{st}$ (кДж/моль)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out = f"qst_vs_loading_{mof}.png"
    plt.savefig(out, dpi=200)
    print(f"  Сохранён: {out}")
    plt.close()


def analyze(csv_file="final_results.csv"):
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(
            f"ОШИБКА: Файл {csv_file} не найден. Сначала запустите main_experiment.py")
        sys.exit(1)

    if "Qst_kJmol" not in data.columns:
        print("ОШИБКА: В CSV нет колонки 'Qst_kJmol'. Убедитесь, что расчёт выполнен с новой версией main_experiment.py")
        sys.exit(1)

    mofs = sorted(data["MOF"].unique())
    print(f"Найдено MOF: {len(mofs)}")

    for mof in mofs:
        print(f"\n-> {mof}")
        plot_isotherms_abs(data, mof)
        plot_isotherms_exc(data, mof)
        plot_qst(data, mof)


if __name__ == "__main__":
    try:
        import pandas
        import matplotlib
    except ImportError:
        print("pip install pandas matplotlib")
        sys.exit(1)

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "final_results.csv"
    analyze(csv_file)
