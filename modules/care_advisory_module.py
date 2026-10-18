import json
from pathlib import Path

BASE=Path(__file__).resolve().parent.parent
CATALOG=json.loads((BASE/"assets"/"disease_catalog.json").read_text(encoding="utf-8"))

PLANTS=["Tomato","Apple","Rose","Grape","Corn"]

GENERIC_STRESS={
    "Nutrient Deficiency":{
        "symptoms":"Yellowing or pale leaves, reduced vigour, weak growth, or older leaves losing colour without a clear disease pattern.",
        "treatment":"Check soil pH and drainage first. Use a balanced plant fertiliser at the label rate, avoid over-fertilising, and monitor new growth for improvement.",
        "medicine":"Balanced Plant Fertiliser",
        "cost":"£6–£15",
        "image":"fertiliser.png",
        "watering":"Keep the soil evenly moist but not waterlogged while the plant recovers. Avoid repeated heavy watering.",
    },
    "Overwatering Stress":{
        "symptoms":"Persistently wet soil, yellowing or soft leaves, drooping despite wet compost, slow growth, or early signs of root stress.",
        "treatment":"Pause watering, empty standing water, improve drainage and airflow, and allow the upper soil layer to begin drying before watering again.",
        "medicine":"No Medicine Suggested",
        "cost":"£0",
        "image":None,
        "watering":"Do not water again until soil moisture falls back into the preferred range for the selected plant.",
    },
    "General Plant Stress":{
        "symptoms":"Wilting, curling, dull colour, slowed growth, or general decline associated with heat, cold, drought, low humidity, or another non-specific environmental stress.",
        "treatment":"Stabilise temperature and moisture, protect the plant from extreme conditions, provide suitable light, and monitor for a more specific disease pattern.",
        "medicine":"Plant Tonic (optional)",
        "cost":"£5–£12",
        "image":"plant_tonic.png",
        "watering":"Correct soil moisture gradually. Avoid both prolonged dryness and sudden overwatering.",
    },
}

LIGHT={
"Tomato":"6–8 hours of direct sunlight daily.",
"Apple":"Full sun, ideally 6–8 hours daily.",
"Rose":"At least 6 hours of sunlight daily.",
"Grape":"Full sun, ideally 7–8 hours daily.",
"Corn":"Full sun, ideally 6–8 hours daily."
}

def plant_diseases(plant):
    # Disease catalogue + the three proposal-specific generic stress conditions.
    return list(CATALOG.get(plant,{}).keys()) + list(GENERIC_STRESS.keys())

def advisory(problem,plant):
    info=GENERIC_STRESS.get(problem) or CATALOG.get(plant,{}).get(problem)
    if info is None:
        info={
          "symptoms":"The image does not match a sufficiently reliable supported condition.",
          "treatment":"Upload a clearer close-up image or seek expert horticultural advice.",
          "medicine":"No Medicine Suggested",
          "cost":"£0",
          "image":None
        }
    fungal=any(k in problem.lower() for k in ["blight","mold","mildew","rot","rust","scab","spot","esca","knot"])
    return {
      "symptoms":info["symptoms"],
      "treatment":info["treatment"],
      "medicine":info["medicine"],
      "cost":info["cost"],
      "medicine_image":str(BASE/"assets"/"medicine"/info["image"]) if info.get("image") else "",
      "watering":info.get("watering") or ("Water at soil level and avoid wetting leaves." if fungal else "Water according to soil moisture and plant needs."),
      "sunlight":info.get("sunlight") or LIGHT.get(plant,"Provide suitable light for the selected plant.")
    }
