from pathlib import Path
import json, subprocess, sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def looks_like_pv(path: Path):
    return path.is_dir() and (path / "Tomato___healthy").is_dir() and (path / "Apple___healthy").is_dir()


def find_plantvillage():
    # Final D-drive layout requested by the project owner.
    d_root = Path(r"D:\Dataset\PlantVillage")

    if d_root.exists():
        direct_candidates = [
            d_root / "raw" / "color",
            d_root / "PlantVillage-Dataset-master" / "raw" / "color",
            d_root / "PlantVillage-Dataset-master (1)" / "PlantVillage-Dataset-master" / "raw" / "color",
        ]
        for p in direct_candidates:
            if looks_like_pv(p):
                return p

        # Handles any extra wrapper folders created by ZIP extraction.
        for p in d_root.rglob("color"):
            try:
                if p.parent.name.lower() == "raw" and looks_like_pv(p):
                    return p
            except OSError:
                pass

    # Fallback for older setups.
    candidates = [
        Path.home() / "Downloads" / "PlantVillage-Dataset-master (1)" / "PlantVillage-Dataset-master" / "raw" / "color",
        Path.home() / "Downloads" / "PlantVillage-Dataset-master" / "PlantVillage-Dataset-master" / "raw" / "color",
        Path.home() / "Downloads" / "PlantVillage-Dataset-master" / "raw" / "color",
    ]
    for p in candidates:
        if looks_like_pv(p):
            return p

    return None


def rose_ready():
    root = HERE / "external" / "rose"
    if not root.exists():
        return False, []
    class_dirs = [p.name for p in root.iterdir() if p.is_dir() and any(x.is_file() for x in p.rglob("*"))]
    return len(class_dirs) >= 2, sorted(class_dirs)


def main():
    pv = find_plantvillage()
    if not pv:
        print("ERROR: PlantVillage raw/color folder was not found automatically.")
        print(r"Expected PlantVillage under D:\Dataset\PlantVillage (or the older Downloads location).")
        print("Current user home:", Path.home())
        print("\nIf needed, run manually:")
        print('python training\\prepare_datasets.py --plantvillage "FULL_PATH_TO_RAW_COLOR"')
        return 2

    print("PlantVillage found:")
    print(pv)
    print("\nPreparing train / validation / test datasets...")
    cmd = [sys.executable, str(HERE / "prepare_datasets.py"), "--plantvillage", str(pv)]
    code = subprocess.call(cmd, cwd=str(PROJECT))
    if code != 0:
        return code

    ready, classes = rose_ready()
    if ready:
        print("\nRose external dataset detected:", ", ".join(classes))
    else:
        print("\nROSE NOTE:")
        print("Rose is not part of PlantVillage. Rose CNN will be skipped until")
        print("rose class folders are added under training\\external\\rose\\<ClassName>\\")

    summary_file = HERE / "dataset_summary.json"
    target_file = HERE / "target_diseases.json"

    if not summary_file.exists():
        print("ERROR: dataset summary was not created.")
        return 4

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    targets = json.loads(target_file.read_text(encoding="utf-8"))

    print("\nPrepared classes:")
    missing_any = False
    for plant in ["tomato", "apple", "grape", "corn", "rose"]:
        classes = list(summary.get(plant, {}).keys())
        print(f"  {plant.title():7s}: {len(classes):2d} class(es)" + (" -> " + ", ".join(classes) if classes else ""))

        missing = [c for c in targets.get(plant, []) if c not in summary.get(plant, {})]
        if missing:
            missing_any = True
            print("    MISSING:", ", ".join(missing))

    if missing_any:
        print("\nERROR: Final training will not start because required classes are missing.")
        return 5

    print("\nAll final CNN target classes are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
