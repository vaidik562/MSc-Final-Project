from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model"
TRAIN = ROOT / "training"
plants = ["tomato", "apple", "grape", "corn", "rose"]

print("=" * 72)
print("GREENGUARDIAN AI - CNN STATUS")
print("=" * 72)
for p in plants:
    m = MODEL / f"{p}.keras"
    c = MODEL / f"{p}_classes.json"
    met = MODEL / f"{p}_metrics.json"
    if m.exists() and c.exists():
        classes = json.loads(c.read_text(encoding="utf-8"))
        msg = f"READY  | {len(classes)} classes"
        if met.exists():
            d = json.loads(met.read_text(encoding="utf-8"))
            msg += f" | accuracy {float(d.get('test_accuracy',0))*100:.2f}% | macro F1 {float(d.get('macro_f1',0))*100:.2f}%"
        print(f"{p.title():7s}: {msg}")
    else:
        print(f"{p.title():7s}: NOT TRAINED")

print("\nEvaluation outputs:", TRAIN / "results")
print("Model outputs     :", MODEL)
