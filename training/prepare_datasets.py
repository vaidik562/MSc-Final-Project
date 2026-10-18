
from pathlib import Path
import argparse, json, random, shutil, re, hashlib, os

HERE=Path(__file__).resolve().parent
PV_MAP=json.loads((HERE/"plantvillage_mapping.json").read_text(encoding="utf-8"))
TARGET=json.loads((HERE/"target_diseases.json").read_text(encoding="utf-8"))
EXTS={".jpg",".jpeg",".png",".bmp",".webp"}

def clean_label(name):
    name=name.strip().replace(" ","_").replace("-","_")
    name=re.sub(r"[^A-Za-z0-9_()]+","_",name)
    return re.sub(r"_+","_",name).strip("_")

def class_seed(label,base_seed):
    return base_seed + int(hashlib.md5(label.encode()).hexdigest()[:6],16)

def _dedupe_by_content(files):
    unique=[]
    seen=set()
    for p in dict.fromkeys(files):
        try:
            h=hashlib.sha1()
            with p.open("rb") as f:
                while True:
                    chunk=f.read(1024*1024)
                    if not chunk: break
                    h.update(chunk)
            digest=h.hexdigest()
        except Exception:
            digest=str(p.resolve())
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(p)
    return unique

def copy_split(files,base,label,seed):
    files=_dedupe_by_content(files)
    random.Random(seed).shuffle(files)
    n=len(files); ntrain=int(n*.70); nval=int(n*.15)
    groups={"train":files[:ntrain],"validation":files[ntrain:ntrain+nval],"test":files[ntrain+nval:]}
    counts={}
    for split,items in groups.items():
        dest=base/split/label
        dest.mkdir(parents=True,exist_ok=True)
        for i,p in enumerate(items):
            target=dest/f"{p.stem}_{i}{p.suffix.lower()}"
            try:
                os.link(p,target)
            except Exception:
                shutil.copy2(p,target)
        counts[split]=len(items)
    return counts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plantvillage",required=True,help="Path to PlantVillage raw/color folder")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    pv=Path(args.plantvillage)
    if not pv.exists(): raise SystemExit(f"PlantVillage color folder not found: {pv}")

    dataset=HERE/"dataset"
    if dataset.exists(): shutil.rmtree(dataset)
    dataset.mkdir()

    summary={}
    for plant in TARGET:
        pool={}
        for src_name,label in PV_MAP.get(plant,{}).items():
            folder=pv/src_name
            if folder.exists():
                pool.setdefault(label,[]).extend(p for p in folder.iterdir() if p.suffix.lower() in EXTS)

        ext_root=HERE/"external"/plant
        if ext_root.exists():
            for disease_dir in (p for p in ext_root.iterdir() if p.is_dir()):
                label=clean_label(disease_dir.name)
                pool.setdefault(label,[]).extend(
                    p for p in disease_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
                )

        summary[plant]={}
        print(f"\n{plant.upper()}")
        for label,files in sorted(pool.items()):
            if not files: continue
            counts=copy_split(files,dataset/plant,label,class_seed(label,args.seed))
            summary[plant][label]={"total":len(files),**counts}

        for disease in TARGET[plant]:
            if disease in summary[plant]:
                print(f"  OK      {disease}: {summary[plant][disease]['total']}")
            else:
                print(f"  MISSING {disease}")

    (HERE/"dataset_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nPrepared dataset:",dataset)
    print("Add missing classes under training/external/<plant>/<disease>/ and rerun this command.")

if __name__=="__main__":
    main()
