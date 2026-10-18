from pathlib import Path
import shutil
import webbrowser
import zipfile

from external_dataset_common import (
    SKIP_TOKENS, norm, image_files, hardlink_or_copy,
    safe_extract, unique_name, open_page
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "external" / "rose"
D_ROSE = Path(r"D:\Dataset\Rose")
WORK = Path(r"D:\GreenGuardian_Project\_dataset_work\Rose")

PAGE = "https://data.mendeley.com/datasets/8jtfk9szbg/3"

ALIASES = {
    "Healthy_Plant": ("fresh leaves", "fresh leaf", "healthy rose", "healthy"),
    "Black_Spot": ("black spot", "blackspot"),
    "Downy_Mildew": ("downy mildew", "downey mildew"),
    "Powdery_Mildew": ("powdery mildew",),
    "Rose_Mosaic": ("rose mosaic", "mosaic"),
    "Rose_Rust": ("rose rust",),
    "Insect_Damage": ("insect hole", "insect infected", "insect infested", "leaf hole"),
}
REQUIRED = set(ALIASES)

def classify(path):
    text = " / ".join(norm(x) for x in Path(path).parts[:-1])
    if any(token in text for token in SKIP_TOKENS):
        return None

    order = [
        "Black_Spot",
        "Downy_Mildew",
        "Powdery_Mildew",
        "Rose_Mosaic",
        "Rose_Rust",
        "Insect_Damage",
        "Healthy_Plant",
    ]
    for label in order:
        if any(alias in text for alias in ALIASES[label]):
            return label
    return None

def signature(root):
    labels = set()
    for p in image_files(root):
        label = classify(p.relative_to(root))
        if label:
            labels.add(label)
    return labels

def import_dataset(source):
    if OUT.exists():
        shutil.rmtree(OUT)
    counts = {k: 0 for k in ALIASES}

    for p in image_files(source):
        rel = p.relative_to(source)
        label = classify(rel)
        if not label:
            continue
        hardlink_or_copy(p, OUT / label / unique_name(rel))
        counts[label] += 1

    print("\nROSE DATASET")
    for label, count in counts.items():
        print(f"  {'OK' if count else 'MISSING':8s} {label:22s}: {count}")

    missing = sorted(REQUIRED - {k for k, v in counts.items() if v > 0})
    if missing:
        print("\nERROR: Rose dataset is incomplete for:")
        for item in missing:
            print(" -", item)
        return False
    return True

def valid_zip():
    if not D_ROSE.exists():
        return None
    for zp in sorted(D_ROSE.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        probe = D_ROSE / "_probe"
        try:
            safe_extract(zp, probe)
            sig = signature(probe)
            shutil.rmtree(probe, ignore_errors=True)
            if REQUIRED.issubset(sig):
                return zp
        except Exception:
            shutil.rmtree(probe, ignore_errors=True)
    return None

def main():
    D_ROSE.mkdir(parents=True, exist_ok=True)

    # Prefer already-extracted local Rose data.
    if image_files(D_ROSE) and REQUIRED.issubset(signature(D_ROSE)):
        print("Using extracted Rose dataset:", D_ROSE)
        return 0 if import_dataset(D_ROSE) else 4

    zp = valid_zip()
    if zp is not None:
        print("Using validated Rose ZIP:", zp)
        safe_extract(zp, WORK)
        return 0 if import_dataset(WORK) else 5

    print("\nRose dataset not found locally.")
    print(r"Required location: D:\Dataset\Rose")
    open_page(PAGE)
    print("Download All from the official page and save/extract it inside D:\\Dataset\\Rose.")
    print("Then run TRAIN_CNN_D_DRIVE.bat again.")
    return 3

if __name__ == "__main__":
    raise SystemExit(main())
