from pathlib import Path
import shutil
import zipfile

from external_dataset_common import (
    norm, image_files, hardlink_or_copy, safe_extract,
    curl_download, open_page, unique_name
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "external" / "grape"
D_GRAPE = Path(r"D:\Dataset\Grape")
WORK = Path(r"D:\GreenGuardian_Project\_dataset_work\GLVD")

PAGE = "https://zenodo.org/records/18937397"
ZIP_URL = (
    "https://zenodo.org/records/18937397/files/"
    "Grapevine%20Leaf%20Variety%20%26%20Disease%20Dataset%20%28GLVD%29.zip?download=1"
)

ALIASES = {
    "Downy_Mildew": ("downy mildew",),
    "Powdery_Mildew": ("powdery mildew",),
    "Bacterial_Rot": ("bacterial rot",),
}
REQUIRED = set(ALIASES)

def classify(path):
    text = " / ".join(norm(x) for x in Path(path).parts[:-1])
    for label, aliases in ALIASES.items():
        if any(alias in text for alias in aliases):
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

    print("\nGRAPE GLVD DATASET")
    for label, count in counts.items():
        print(f"  {'OK' if count else 'MISSING':8s} {label:22s}: {count}")

    missing = sorted(REQUIRED - {k for k, v in counts.items() if v > 0})
    if missing:
        print("\nERROR: GLVD is incomplete for:")
        for item in missing:
            print(" -", item)
        return False
    return True

def find_valid_zip():
    if not D_GRAPE.exists():
        return None
    for zp in sorted(D_GRAPE.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        probe = D_GRAPE / "_probe"
        try:
            safe_extract(zp, probe)
            ok = REQUIRED.issubset(signature(probe))
            shutil.rmtree(probe, ignore_errors=True)
            if ok:
                return zp
        except Exception:
            shutil.rmtree(probe, ignore_errors=True)
    return None

def main():
    D_GRAPE.mkdir(parents=True, exist_ok=True)

    # Already-extracted GLVD folder.
    if image_files(D_GRAPE) and REQUIRED.issubset(signature(D_GRAPE)):
        print("Using extracted GLVD:", D_GRAPE)
        return 0 if import_dataset(D_GRAPE) else 4

    zp = find_valid_zip()
    if zp is None:
        zp = curl_download(
            ZIP_URL,
            D_GRAPE / "Grapevine_Leaf_Variety_and_Disease_Dataset_GLVD.zip",
            "Zenodo GLVD grape dataset (about 1.1 GB)",
        )

    if zp is None or not zipfile.is_zipfile(zp):
        open_page(PAGE)
        print(r"Download the GLVD ZIP and save it in D:\Dataset\Grape, then rerun.")
        return 3

    safe_extract(zp, WORK)
    if not REQUIRED.issubset(signature(WORK)):
        print("ERROR: Grape ZIP does not contain the expected GLVD disease classes.")
        return 5

    return 0 if import_dataset(WORK) else 6

if __name__ == "__main__":
    raise SystemExit(main())
