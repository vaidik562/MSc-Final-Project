import numpy as np

GENERIC_STRESS_CONDITIONS = [
    "Nutrient Deficiency",
    "Overwatering Stress",
    "General Plant Stress",
]


def _leaf_colour_signals(image):
    """Return lightweight colour signals from the central image area.

    This is deliberately a transparent heuristic, not a CNN disease classifier.
    It is used only for the proposal's generic stress categories when a trained
    disease class is not already confidently selected.
    """
    arr = np.asarray(image.convert("RGB").resize((180, 180)), dtype=np.float32) / 255.0
    # Reduce influence from the outer background by focusing on the central 80%.
    pad = 18
    arr = arr[pad:-pad, pad:-pad]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green = (g > 0.20) & (g >= r * 0.90) & (g >= b * 1.03)
    yellow = (
        (r > 0.42) & (g > 0.38) & (b < 0.42) &
        (r > b * 1.22) & (g > b * 1.20) & (np.abs(r - g) < 0.28)
    )
    brown = (r > 0.24) & (g > 0.12) & (g < r * 0.95) & (b < g * 0.98)
    vegetation = green | yellow | brown
    veg_count = max(int(vegetation.sum()), 1)

    return {
        "green_ratio": float(green.sum() / veg_count),
        "yellow_ratio": float(yellow.sum() / veg_count),
        "brown_ratio": float(brown.sum() / veg_count),
        "vegetation_fraction": float(vegetation.mean()),
    }


def resolve_condition(image, base_problem, base_confidence, base_method, plant, soil, last_days, humidity, temp):
    """Combine disease output with proposal-specific stress screening.

    Priority:
    1. A confident specific disease from the classifier/prototype is preserved.
    2. Healthy/uncertain results may be replaced by a generic stress condition
       when environmental and/or colour evidence is sufficiently strong.
    3. Otherwise the base result is returned unchanged.

    The returned method clearly says Rule-based Stress Assessment whenever this
    module supplies the final condition so it is never presented as CNN output.
    """
    soil = float(soil)
    humidity = float(humidity)
    temp = float(temp)
    last_days = int(last_days)
    conf = float(base_confidence or 0.0)

    # Preserve specific disease classifications. Generic stress rules should not
    # overwrite a supported disease diagnosis.
    if base_problem not in {"Healthy Plant", "__UNCERTAIN__", None}:
        return base_problem, conf, base_method

    signals = _leaf_colour_signals(image)
    yellow = signals["yellow_ratio"]

    # Overwatering: the measured soil/watering inputs are the primary evidence.
    overwater_score = 0
    if soil >= 85:
        overwater_score += 3
    elif soil >= 75:
        overwater_score += 2
    elif soil >= 68:
        overwater_score += 1
    if last_days <= 1 and soil >= 70:
        overwater_score += 1
    if humidity >= 80 and soil >= 70:
        overwater_score += 1
    if yellow >= 0.12 and soil >= 70:
        overwater_score += 1

    if overwater_score >= 3:
        stress_conf = min(94.0, 70.0 + overwater_score * 4.0 + min(yellow, 0.25) * 20.0)
        return "Overwatering Stress", round(stress_conf, 1), "Rule-based Stress Assessment"

    # Nutrient deficiency: visual yellowing is required; environmental values
    # help avoid labelling obvious wet/dry stress as nutrient deficiency.
    balanced_moisture = 25 <= soil <= 72
    if yellow >= 0.22 and balanced_moisture:
        nutrient_score = 2
        if yellow >= 0.34:
            nutrient_score += 1
        if 12 <= temp <= 32:
            nutrient_score += 1
        if 30 <= humidity <= 80:
            nutrient_score += 1
        stress_conf = min(92.0, 66.0 + nutrient_score * 4.5 + min(yellow, 0.45) * 18.0)
        return "Nutrient Deficiency", round(stress_conf, 1), "Rule-based Stress Assessment"

    # General plant stress covers environmental extremes not represented by a
    # specific disease class (e.g. heat/cold/drought stress).
    stress_points = 0
    if temp >= 35 or temp <= 7:
        stress_points += 2
    elif temp >= 32 or temp <= 10:
        stress_points += 1
    if soil <= 15:
        stress_points += 2
    elif soil <= 22:
        stress_points += 1
    if last_days >= 5 and soil <= 30:
        stress_points += 1
    if humidity <= 25:
        stress_points += 1
    if yellow >= 0.14:
        stress_points += 1

    if stress_points >= 3:
        stress_conf = min(91.0, 64.0 + stress_points * 5.0)
        return "General Plant Stress", round(stress_conf, 1), "Rule-based Stress Assessment"

    return base_problem, conf, base_method
