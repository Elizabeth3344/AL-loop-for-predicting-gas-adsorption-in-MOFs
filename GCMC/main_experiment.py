import os
import subprocess
import shutil
import sys
import glob
import csv
import math

from compute_void_fraction import compute_void_fractions

# ================= НАСТРОЙКИ ЭКСПЕРИМЕНТА =================
GAS_NAME = "CO2"
TEMPERATURES = [298]                                        # Кельвины
PRESSURES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]  # Бары
DEFAULT_VOID_FRACTION = 0.79  # Используется, если Zeo++ недоступен
CIF_FILTER = None             # None — все .cif; ["MOF5.cif"] — только указанные
OUTPUT_CSV = "final_results_selected10.csv"
# ==========================================================

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIFS_DIR = os.path.join(BASE_DIR, "selected_10_cifs")
ZEOPP_BIN = os.path.join(BASE_DIR, "zeo++-0.3", "network")
RASPA_DIR = os.environ.get("RASPA_DIR")
SIMULATE_BIN = os.path.join(
    RASPA_DIR, "bin", "simulate") if RASPA_DIR else None


def check_setup():
    if not RASPA_DIR:
        print("ОШИБКА: Переменная окружения RASPA_DIR не установлена.")
        print("  Пример: export RASPA_DIR=/usr/local  (если simulate лежит в /usr/local/bin/simulate)")
        sys.exit(1)
    if not os.path.exists(SIMULATE_BIN):
        print(f"ОШИБКА: Бинарник RASPA не найден: {SIMULATE_BIN}")
        print(f"  RASPA_DIR={RASPA_DIR}")
        print("  Найдите правильный путь командой: find / -name simulate 2>/dev/null")
        sys.exit(1)
    if not os.path.isdir(CIFS_DIR):
        print(f"ОШИБКА: Папка {CIFS_DIR} не найдена.")
        sys.exit(1)


def get_unit_cells(cif_path, cutoff=12.0):
    """Возвращает строку 'na nb nc' так, чтобы каждая сторона бокса >= 2*cutoff."""
    a, b, c = 1.0, 1.0, 1.0
    with open(cif_path) as f:
        for line in f:
            l = line.strip()
            if l.startswith("_cell_length_a"):
                a = float(l.split()[-1])
            elif l.startswith("_cell_length_b"):
                b = float(l.split()[-1])
            elif l.startswith("_cell_length_c"):
                c = float(l.split()[-1])
    na = max(1, math.ceil(2 * cutoff / a))
    nb = max(1, math.ceil(2 * cutoff / b))
    nc = max(1, math.ceil(2 * cutoff / c))
    return f"{na} {nb} {nc}"


def parse_output(directory, mof_name, temp, press):
    """Читает результаты из файла .data"""
    out_dir = os.path.join(directory, "Output", "System_0")
    files = glob.glob(os.path.join(out_dir, "*.data"))

    loading_abs = 0.0
    loading_exc = 0.0
    heat = 0.0

    if files:
        with open(files[0], 'r') as f:
            for line in f:
                if "Average loading absolute [mol/kg framework]" in line:
                    try:
                        loading_abs = float(line.split()[5])
                    except:
                        pass
                elif "Average loading excess [mol/kg framework]" in line:
                    try:
                        loading_exc = float(line.split()[5])
                    except:
                        pass
                elif "Average  <U_gh>_1-<U_h>_0:" in line:
                    try:
                        # Формат: [CO2] Average  <U_gh>_1-<U_h>_0: VALUE +/- ERR [K]  (KJ_VALUE +/- ERR kJ/mol)
                        kjmol_part = line.split("(")[1]
                        u_gh_kJmol = float(kjmol_part.split()[0])
                        # Q_st = -(U_gh - U_h) + RT  (кинетическая энергия одинакова в газе и адсорбате → сокращается)
                        R = 8.314e-3  # kJ/(mol·K)
                        heat = -u_gh_kJmol + R * temp
                    except:
                        pass

    return {
        "MOF": mof_name,
        "T": temp,
        "P": press,
        "Abs_mol_kg": loading_abs,
        "Exc_mol_kg": loading_exc,
        "Qst_kJmol": heat
    }


def run_isotherms(void_fractions, zeopp_available=True):
    print("\n--- ЭТАП: Расчет изотерм адсорбции ---")
    results = []

    with open(os.path.join(BASE_DIR, "simulation.input.template"), 'r') as f:
        template_content = f.read()

    cif_files = sorted(glob.glob(os.path.join(CIFS_DIR, "*.cif")))
    if CIF_FILTER is not None:
        cif_files = [f for f in cif_files if os.path.basename(f) in CIF_FILTER]
    print(f"Найдено MOF: {len(cif_files)}")

    for cif_path in cif_files:
        mof_name = os.path.splitext(os.path.basename(cif_path))[0]
        vf = void_fractions.get(mof_name)
        if vf is None:
            if zeopp_available:
                # Zeo++ запускался, но упал для этого конкретного MOF
                print(
                    f"  ПРЕДУПРЕЖДЕНИЕ: Zeo++ не вернул VF для {mof_name}, используем {DEFAULT_VOID_FRACTION}")
            vf = DEFAULT_VOID_FRACTION

        uc = get_unit_cells(cif_path)
        print(f"\n-> MOF: {mof_name}  (VF={vf:.4f}, UnitCells={uc})")

        # Подготовка структуры → simulation_structure.cif в BASE_DIR
        subprocess.call(
            [sys.executable,
             os.path.join(BASE_DIR, "convert_to_raspa.py"),
             cif_path, GAS_NAME],
            cwd=BASE_DIR
        )

        for temp in TEMPERATURES:
            for press_bar in PRESSURES:
                press_pa = press_bar * 100000.0
                print(f"   -> T={temp}K, P={press_bar} bar...")

                point_dir = os.path.join(
                    BASE_DIR, "Results", mof_name, f"{temp}K", f"{press_bar}bar")
                os.makedirs(point_dir, exist_ok=True)

                shutil.copy(os.path.join(
                    BASE_DIR, "simulation_structure.cif"), point_dir)
                shutil.copy(os.path.join(
                    BASE_DIR, f"{GAS_NAME}.def"), point_dir)
                for ff_file in glob.glob(os.path.join(BASE_DIR, "ForceField", "*.def")):
                    shutil.copy(ff_file, point_dir)

                movie_flag = "yes" if press_bar == PRESSURES[0] else "no"
                sim_content = (template_content
                               .replace("TEMP_PLACEHOLDER", str(temp))
                               .replace("PRES_PLACEHOLDER", str(press_pa))
                               .replace("REPLACE_ME", GAS_NAME)
                               .replace("MOVIE_FLAG", movie_flag)
                               .replace("HVF_PLACEHOLDER", str(vf))
                               .replace("UC_PLACEHOLDER", uc))

                with open(os.path.join(point_dir, "simulation.input"), 'w') as f:
                    f.write(sim_content)

                subprocess.run([SIMULATE_BIN, "simulation.input"],
                               cwd=point_dir, stdout=subprocess.DEVNULL)

                res = parse_output(point_dir, mof_name, temp, press_bar)
                results.append(res)

    return results


def save_csv(results):
    filename = os.path.join(BASE_DIR, OUTPUT_CSV)
    print(f"\n--- Сохранение результатов в {filename} ---")
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=["MOF", "T", "P", "Abs_mol_kg", "Exc_mol_kg", "Qst_kJmol"])
        writer.writeheader()
        writer.writerows(results)
    print("Готово!")


if __name__ == "__main__":
    check_setup()

    # Шаг 1: Вычисляем Helium Void Fraction для всех MOF
    print("--- ЭТАП: Расчет Helium Void Fraction (Zeo++) ---")
    if os.path.exists(ZEOPP_BIN):
        void_fractions = compute_void_fractions(CIFS_DIR, ZEOPP_BIN)
        zeopp_available = True
    else:
        print(f"\n{'='*60}")
        print(f"ПРЕДУПРЕЖДЕНИЕ: Zeo++ не найден по пути:")
        print(f"  {ZEOPP_BIN}")
        print(
            f"Для ВСЕХ {len(glob.glob(os.path.join(CIFS_DIR, '*.cif')))} MOF будет использовано")
        print(
            f"значение по умолчанию: HeliumVoidFraction = {DEFAULT_VOID_FRACTION}")
        print(f"Это может снизить точность расчёта абсолютной загрузки.")
        print(f"{'='*60}\n")
        void_fractions = {}
        zeopp_available = False

    # Шаг 2: Запускаем изотермы
    data = run_isotherms(void_fractions, zeopp_available)
    save_csv(data)
