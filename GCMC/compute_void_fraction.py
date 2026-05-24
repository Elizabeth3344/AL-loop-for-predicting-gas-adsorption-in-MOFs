import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CIFS_DIR = os.path.join(BASE_DIR, "selected_10_cifs")
DEFAULT_ZEOPP = os.path.join(BASE_DIR, "zeo++-0.3", "network")
PROBE_RADIUS = 1.29  # радиус гелия (Å)
VOL_SAMPLES = 50000
FALLBACK_VF = 0.79  # значение по умолчанию если Zeo++ не справился


def compute_void_fractions(cifs_dir=DEFAULT_CIFS_DIR, zeopp_path=DEFAULT_ZEOPP):
    """
    Вычисляет Helium Void Fraction для каждого CIF-файла в cifs_dir с помощью Zeo++.
    Возвращает dict: {mof_name: void_fraction}  (None если ошибка)
    """
    results = {}
    for cif in sorted(os.listdir(cifs_dir)):
        if not cif.endswith(".cif"):
            continue
        name = cif.replace(".cif", "")
        cif_path = os.path.join(cifs_dir, cif)
        vol_file = os.path.join(cifs_dir, f"{name}.vol")

        try:
            # Удаляем старый файл, чтобы не прочитать устаревшие данные
            if os.path.exists(vol_file):
                os.remove(vol_file)

            subprocess.run(
                [zeopp_path, "-vol",
                 str(PROBE_RADIUS), str(PROBE_RADIUS), str(VOL_SAMPLES),
                 cif_path],
                check=True, capture_output=True, timeout=120
            )

            if not os.path.exists(vol_file):
                print(f"  {name}: ОШИБКА - Zeo++ не создал выходной файл")
                results[name] = None
                continue

            with open(vol_file) as f:
                content = f.read()

            vf = None
            if "AV_Volume_fraction:" in content:
                idx = content.index("AV_Volume_fraction:")
                vf = float(content[idx:].split()[1])

            if vf is not None:
                print(f"  {name}: VF = {vf:.4f}")
                results[name] = vf
            else:
                print(f"  {name}: VF не найден, используем {FALLBACK_VF}. Содержимое: {content[:200]!r}")
                results[name] = FALLBACK_VF
        except Exception as e:
            print(f"  {name}: Zeo++ не справился, используем VF = {FALLBACK_VF} (причина: {type(e).__name__})")
            results[name] = FALLBACK_VF

    return results


if __name__ == "__main__":
    print("=== Расчёт Helium Void Fraction (Zeo++) ===")
    vfs = compute_void_fractions()
    print("\n=== Результаты ===")
    for name, vf in vfs.items():
        print(f"{name}: HeliumVoidFraction {vf}")
