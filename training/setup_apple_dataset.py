from pathlib import Path
import shutil
import zipfile

from external_dataset_common import (
    SKIP_TOKENS, norm, image_files, hardlink_or_copy,
    safe_extract, curl_download, open_page, unique_name
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "external" / "apple"
D_APPLE = Path(r"D:\Dataset\Apple")
WORK = Path(r"D:\GreenGuardian_Project\_dataset_work\AppleLeaf9")

PAGE = "https://github.com/JasonYangCode/AppleLeaf9"
ZIP_URL = "https://codeload.github.com/JasonYangCode/AppleLeaf9/zip/refs/heads/main"

ALIASES = {
    "Healthy_Plant": ("health", "healthy"),
    "Apple_Scab": ("scab",),
    "Alternaria_Leaf_Spot": ("alternaria leaf spot", "alternaria"),
    "Brown_Spot": ("brown spot",),
    "Frogeye_Leaf_Spot": ("frogeye leaf spot", "frog eye leaf spot", "frogeye"),
    "Grey_Spot": ("grey spot", "gray spot"),
    "Mosaic": ("mosaic",),
    "Powdery_Mildew": ("powdery mildew",),
}

REQUIRED_EXTERNAL = {
    "Alternaria_Leaf_Spot",
    "Brown_Spot",
    "Frogeye_Leaf_Spot",
    "Grey_Spot",
    "Mosaic",
    "Powdery_Mildew",
}

def classify(path):
    text = " / ".join(norm(x) for x in Path(path).parts[:-1])
    if any(token in text for token in SKIP_TOKENS):
        return None

    order = [
        "Alternaria_Leaf_Spot",
        "Frogeye_Leaf_Spot",
        "Powdery_Mildew",
        "Brown_Spot",
        "Grey_Spot",
        "Apple_Scab",
        "Mosaic",
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

    print("\nAPPLE DATASET")
    for label, count in counts.items():
        print(f"  {'OK' if count else 'MISSING':8s} {label:22s}: {count}")

    missing = sorted(REQUIRED_EXTERNAL - {k for k, v in counts.items() if v > 0})
    if missing:
        print("\nERROR: AppleLeaf9 is incomplete for:")
        for item in missing:
            print(" -", item)
        return False
    return True

def find_valid_zip():
    if not D_APPLE.exists():
        return None
    for zp in sorted(D_APPLE.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        probe = D_APPLE / "_probe"
        try:
            safe_extract(zp, probe)
            ok = REQUIRED_EXTERNAL.issubset(signature(probe))
            shutil.rmtree(probe, ignore_errors=True)
            if ok:
                return zp
        except Exception:
            shutil.rmtree(probe, ignore_errors=True)
    return None

def main():
    D_APPLE.mkdir(parents=True, exist_ok=True)

    # Already-extracted AppleLeaf9 folder.
    if image_files(D_APPLE) and REQUIRED_EXTERNAL.issubset(signature(D_APPLE)):
        print("Using extracted AppleLeaf9:", D_APPLE)
        return 0 if import_dataset(D_APPLE) else 4

    zp = find_valid_zip()
    if zp is None:
        zp = curl_download(ZIP_URL, D_APPLE / "AppleLeaf9-main.zip", "AppleLeaf9")

    if zp is None or not zipfile.is_zipfile(zp):
        open_page(PAGE)
        print(r"Download Code -> Download ZIP and save it in D:\Dataset\Apple, then rerun.")
        return 3

    safe_extract(zp, WORK)
    if not REQUIRED_EXTERNAL.issubset(signature(WORK)):
        print("ERROR: Apple ZIP does not contain the expected AppleLeaf9 classes.")
        return 5

    return 0 if import_dataset(WORK) else 6

if __name__ == "__main__":
    raise SystemExit(main())
