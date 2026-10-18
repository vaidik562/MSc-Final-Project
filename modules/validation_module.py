
import numpy as np

def validate_analysis(plant,image,temp,humidity,soil,last_days):
    errors=[]
    if not plant: errors.append("Please select a plant type.")
    if image is None: errors.append("Please upload an image or take a photo.")
    if not (-5<=temp<=50): errors.append("Temperature must be between -5°C and 50°C.")
    if not (0<=humidity<=100): errors.append("Humidity must be between 0% and 100%.")
    if not (0<=soil<=100): errors.append("Soil moisture must be between 0% and 100%.")
    if last_days<0: errors.append("Last watered days cannot be negative.")
    if image is not None and (image.width<128 or image.height<128):
        errors.append("Please use a clearer image of at least 128×128 pixels.")
    return errors

def leaf_likeness(image):
    """Return a 0..1 *heuristic* centred-foliage score, not a probability.

    Background trees alone must not automatically make a foreground car or
    other large unrelated object pass. Weight the centre of the frame more
    heavily than the edges because the UI asks for one centred, close-up leaf.
    Include brown and yellow foliage as well as green, but avoid counting
    saturated red surfaces as brown leaves.
    """
    rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    green = (g > 0.16) & (g >= r * 0.94) & (g > b * 1.04)
    yellow = (r > 0.32) & (g > 0.30) & (b < 0.55) & (r > b * 1.18) & (g >= r * 0.48)
    brown = (
        (r > 0.19) & (g > 0.10) & (r > g * 1.07)
        & (g >= r * 0.38) & (b < g * 1.02)
    )
    foliage = green | yellow | brown

    whole = float(foliage.mean())
    mid = float(foliage[64:192, 64:192].mean())
    inner = float(foliage[96:160, 96:160].mean())
    return float(0.20 * whole + 0.35 * mid + 0.45 * inner)
