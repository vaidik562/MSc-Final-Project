from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import urllib.request
import webbrowser
import zipfile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SKIP_TOKENS = ("augment", "augmented", "augmentation", "synthetic", "generated")

def norm(value):
    value = str(value).lower().replace("_", " ").replace("-", " ")
    return " ".join(value.split())

def image_files(root):
    root = Path(root)
    if not root.exists():
        return []
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

def hardlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def safe_extract(zip_path, dest):
    zip_path = Path(zip_path)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    base = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError("Unsafe ZIP path detected.")
        z.extractall(dest)
    return dest

def curl_download(url, target, title):
    """Resume-capable Windows download; urllib fallback if curl.exe is unavailable."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {title}...")
    print("Destination:", target)

    try:
        result = subprocess.run(
            [
                "curl.exe", "-L", "--fail",
                "--retry", "5", "--retry-delay", "5",
                "-C", "-", "-o", str(target), url,
            ],
            check=False,
        )
        if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return target
    except FileNotFoundError:
        pass

    print("curl.exe unavailable/failed; trying Python downloader...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 GreenGuardianAI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response, target.open("wb") as f:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r  {received * 100 / total:5.1f}%", end="", flush=True)
        print()
        return target if target.exists() and target.stat().st_size > 0 else None
    except Exception as exc:
        print("Download failed:", exc)
        return None

def open_page(url):
    print("Opening official dataset page:")
    print(url)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def unique_name(path):
    path = Path(path)
    digest = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{path.stem}_{digest}{path.suffix.lower()}"
