from pathlib import Path
import argparse, json, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
OUT = HERE.parent / "model"
RESULTS = HERE / "results"
TARGETS = json.loads((HERE / "target_diseases.json").read_text(encoding="utf-8"))
IMG = (224, 224)
BATCH = 16
SEED = 42


def _class_weights(train_dir: Path, classes):
    counts = []
    for c in classes:
        n = sum(1 for p in (train_dir / c).rglob("*") if p.is_file())
        counts.append(max(n, 1))
    total = sum(counts)
    k = len(counts)
    return {i: total / (k * c) for i, c in enumerate(counts)}


def _collect_predictions(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_prob.extend(p.tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)


def _choose_rejection_thresholds(y_true, probs):
    if len(y_true) == 0:
        return {"min_confidence": 60.0, "min_margin": 12.0}
    order = np.argsort(probs, axis=1)[:, ::-1]
    top = order[:, 0]
    second = order[:, 1] if probs.shape[1] > 1 else order[:, 0]
    top_prob = probs[np.arange(len(probs)), top] * 100.0
    second_prob = probs[np.arange(len(probs)), second] * 100.0 if probs.shape[1] > 1 else np.zeros(len(probs))
    margin = top_prob - second_prob
    correct = top == y_true
    overall = float(np.mean(correct)) if len(correct) else 0.0
    target_acc = max(0.90, overall)

    best = None
    for conf in (60, 65, 70, 75, 80, 85):
        for mar in (10, 12, 15, 18, 20, 25):
            accept = (top_prob >= conf) & (margin >= mar)
            coverage = float(np.mean(accept))
            if accept.any():
                accepted_acc = float(np.mean(correct[accept]))
            else:
                accepted_acc = 0.0
            if coverage >= 0.55 and accepted_acc >= target_acc:
                score = (coverage, accepted_acc, conf + mar / 100.0)
                if best is None or score > best[0]:
                    best = (score, conf, mar, coverage, accepted_acc)

    if best is None:
        return {"min_confidence": 60.0, "min_margin": 12.0}

    _, conf, mar, coverage, accepted_acc = best
    return {
        "min_confidence": float(conf),
        "min_margin": float(mar),
        "validation_coverage": round(coverage, 4),
        "validation_accepted_accuracy": round(accepted_acc, 4),
    }


def _save_confusion_png(cm, classes, path):
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(max(7, len(classes) * 0.75), max(6, len(classes) * 0.68)))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print("WARNING: could not save confusion-matrix PNG:", exc)


def _save_history_png(history_rows, path):
    try:
        import matplotlib.pyplot as plt
        if not history_rows:
            return
        epochs = list(range(1, len(history_rows) + 1))
        acc = [r.get("accuracy") for r in history_rows]
        val_acc = [r.get("val_accuracy") for r in history_rows]
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(epochs, acc, label="train accuracy")
        ax.plot(epochs, val_acc, label="validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Accuracy")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print("WARNING: could not save training-curve PNG:", exc)


def train_plant(plant, force=False):
    root = DATA / plant
    slug = plant.lower()
    model_path = OUT / f"{slug}.keras"
    classes_path = OUT / f"{slug}_classes.json"

    if model_path.exists() and classes_path.exists() and not force:
        try:
            existing_classes = json.loads(classes_path.read_text(encoding="utf-8"))
        except Exception:
            existing_classes = []

        expected_classes = set(TARGETS.get(slug, []))
        if set(existing_classes) == expected_classes:
            print(f"SKIP {plant}: compatible trained model already exists")
            return True

        print(f"RETRAIN {plant}: existing model classes do not match final target scope")

    if not (root / "train").exists():
        print(f"SKIP {plant}: dataset not prepared")
        return False

    train = tf.keras.utils.image_dataset_from_directory(
        root / "train", image_size=IMG, batch_size=BATCH, seed=SEED, shuffle=True
    )
    val = tf.keras.utils.image_dataset_from_directory(
        root / "validation", image_size=IMG, batch_size=BATCH, seed=SEED, shuffle=False
    )
    test = tf.keras.utils.image_dataset_from_directory(
        root / "test", image_size=IMG, batch_size=BATCH, shuffle=False
    )
    classes = train.class_names

    expected = set(TARGETS.get(slug, []))
    found = set(classes)
    missing = sorted(expected - found)
    if missing:
        print(f"SKIP {plant}: target dataset is incomplete.")
        print("Missing required classes:")
        for name in missing:
            print("  -", name)
        return False

    if len(classes) < 2:
        print(f"SKIP {plant}: need at least 2 classes, found {len(classes)}")
        return False

    auto = tf.data.AUTOTUNE
    train = train.prefetch(auto)
    val = val.prefetch(auto)
    test = test.prefetch(auto)

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inp = keras.Input(shape=(224, 224, 3), name="image")
    x = augment(inp)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(len(classes), activation="softmax", name="prediction")(x)
    model = keras.Model(inp, out, name=f"greenguardian_{slug}_mobilenetv2")

    weights = _class_weights(root / "train", classes)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, min_lr=1e-6),
    ]

    print("\nSTAGE 1/2: training classifier head")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train,
        validation_data=val,
        epochs=10,
        callbacks=callbacks,
        class_weight=weights,
    )

    print("\nSTAGE 2/2: fine-tuning upper MobileNetV2 layers")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train,
        validation_data=val,
        epochs=6,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.3, min_lr=1e-7),
        ],
        class_weight=weights,
    )

    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    model.save(model_path)
    classes_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    print("\nCalibrating rejection thresholds on validation set...")
    val_true, val_prob = _collect_predictions(model, val)
    thresholds = _choose_rejection_thresholds(val_true, val_prob)
    (OUT / f"{slug}_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print("Evaluating held-out test set...")
    test_true, test_prob = _collect_predictions(model, test)
    test_pred = np.argmax(test_prob, axis=1)
    report = classification_report(
        test_true,
        test_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_true, test_pred, labels=list(range(len(classes))))
    accuracy = float(np.mean(test_pred == test_true)) if len(test_true) else 0.0

    history_rows = []
    for history in (h1.history, h2.history):
        n = max(len(v) for v in history.values()) if history else 0
        for i in range(n):
            row = {k: float(v[i]) for k, v in history.items() if i < len(v)}
            history_rows.append(row)

    metrics = {
        "plant": plant,
        "test_accuracy": accuracy,
        "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "classes": classes,
        "rejection_thresholds": thresholds,
    }
    (OUT / f"{slug}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS / f"{slug}_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(RESULTS / f"{slug}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (RESULTS / f"{slug}_training_history.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
    _save_confusion_png(cm, classes, RESULTS / f"{slug}_confusion_matrix.png")
    _save_history_png(history_rows, RESULTS / f"{slug}_training_curve.png")

    print("\n" + "=" * 68)
    print(f"{plant.upper()} COMPLETE")
    print(f"Test accuracy : {accuracy * 100:.2f}%")
    print(f"Macro F1      : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Model         : {model_path}")
    print(f"Thresholds    : {thresholds}")
    print("=" * 68)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="all", choices=["all", "tomato", "apple", "rose", "grape", "corn"])
    ap.add_argument("--force", action="store_true", help="Retrain even if model files already exist")
    args = ap.parse_args()
    plants = ["tomato", "apple", "grape", "corn", "rose"] if args.plant == "all" else [args.plant]
    completed = []
    skipped = []
    for p in plants:
        print("\n" + "#" * 72 + f"\nTRAINING {p.upper()}\n" + "#" * 72)
        ok = train_plant(p, force=args.force)
        (completed if ok else skipped).append(p)
    print("\nTRAINING SUMMARY")
    print("Completed/available:", ", ".join(completed) if completed else "none")
    print("Skipped/missing    :", ", ".join(skipped) if skipped else "none")

    if skipped:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
