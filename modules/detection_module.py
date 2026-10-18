
from pathlib import Path
from functools import lru_cache
import json, re
import numpy as np
from PIL import ImageStat

BASE=Path(__file__).resolve().parent.parent
MODEL_DIR=BASE/"model"

SLUG={"Tomato":"tomato","Apple":"apple","Rose":"rose","Grape":"grape","Corn":"corn"}

def _paths(plant):
    slug=SLUG[plant]
    return MODEL_DIR/f"{slug}.keras", MODEL_DIR/f"{slug}_classes.json"

def model_available(plant):
    m,c=_paths(plant)
    return m.exists() and c.exists()

def any_model_available():
    return any(model_available(p) for p in SLUG)

@lru_cache(maxsize=5)
def _load(plant):
    import tensorflow as tf
    m,c=_paths(plant)
    model=tf.keras.models.load_model(m)
    classes=json.loads(c.read_text(encoding="utf-8"))
    return model,classes

def friendly(raw):
    return str(raw).replace("_"," ").replace("  "," ").strip()

def predict_cnn(image,plant):
    if not model_available(plant):
        return None,0.0,"Training pending"

    model,classes=_load(plant)
    arr=np.array(image.convert("RGB").resize((224,224)),dtype="float32")
    # Model contains MobileNetV2 preprocess_input internally; feed raw 0..255 RGB.
    probs=np.asarray(model.predict(arr[None,...],verbose=0)[0],dtype=float)
    order=np.argsort(probs)[::-1]
    top=int(order[0]); second=int(order[1]) if len(order)>1 else top
    conf=float(probs[top])*100
    margin=(float(probs[top])-float(probs[second]))*100 if len(order)>1 else 100.0

    min_conf=60.0
    min_margin=12.0
    threshold_path=MODEL_DIR/f"{SLUG[plant]}_thresholds.json"
    if threshold_path.exists():
        try:
            threshold_data=json.loads(threshold_path.read_text(encoding="utf-8"))
            min_conf=float(threshold_data.get("min_confidence",min_conf))
            min_margin=float(threshold_data.get("min_margin",min_margin))
        except Exception:
            pass

    if conf < min_conf or margin < min_margin:
        return "__UNCERTAIN__",round(conf,1),"Trained MobileNetV2 CNN"

    return friendly(classes[top]),round(conf,1),"Trained MobileNetV2 CNN"

def prototype_demo(image,file_name,plant,humidity,temp,soil):
    """
    Transparent fallback used only before a trained plant-specific model is installed.
    It is deliberately labelled as Prototype Demo and must not be presented as CNN output.
    """
    img=image.convert("RGB").resize((224,224))
    stat=ImageStat.Stat(img)
    r,g,b=stat.mean
    brightness=(r+g+b)/3
    greenness=g-((r+b)/2)
    name=(file_name or "").lower()

    if "mildew" in name or "powder" in name:
        problem="Powdery Mildew"
    elif "rust" in name:
        problem=("Rust" if plant=="Apple" else
                 ("Rose Rust" if plant=="Rose" else
                  ("Common Rust" if plant=="Corn" else "__UNCERTAIN__")))
    elif plant=="Corn" and ("gray" in name or "grey" in name) and ("spot" in name or "leaf" in name):
        problem="Gray Leaf Spot"
    elif plant=="Corn" and ("northern" in name or "leaf_blight" in name or "leafblight" in name):
        problem="Northern Leaf Blight"
    elif "scab" in name and plant=="Apple":
        problem="Apple Scab"
    elif "blackspot" in name or "black_spot" in name:
        problem="Black Spot" if plant=="Rose" else "__UNCERTAIN__"
    elif greenness>20 and 30<=soil<=75 and brightness>70:
        problem="Healthy Plant"
    else:
        problem="__UNCERTAIN__"
    return problem,70.0 if problem!="__UNCERTAIN__" else 0.0,"Prototype Demo"

def predict(image,file_name,plant,humidity,temp,soil):
    out=predict_cnn(image,plant)
    if out[0] is not None:
        return out
    return prototype_demo(image,file_name,plant,humidity,temp,soil)
