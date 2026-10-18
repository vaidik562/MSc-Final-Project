def weather_risk(problem,humidity,temp):
    m=[]
    if humidity>=75: m.append("High humidity may increase fungal disease risk and slow soil drying.")
    if humidity<=35: m.append("Low humidity may increase plant water stress.")
    if temp>=30: m.append("High temperature may increase water loss.")
    if temp<=10: m.append("Low temperature may slow plant recovery.")
    if any(k in problem.lower() for k in ["blight","mold","mildew","rot","rust","scab","spot"]) and humidity>=65:
        m.append("The detected condition may spread faster in damp conditions.")
    if problem=="Overwatering Stress" and humidity>=70:
        m.append("Humid conditions may prolong overwatering stress because the soil can dry more slowly.")
    if problem=="General Plant Stress" and (temp>=32 or temp<=10):
        m.append("Temperature stress is contributing to the general plant-stress assessment.")
    if problem=="Nutrient Deficiency" and (temp>=32 or temp<=10):
        m.append("Extreme temperature can reduce nutrient uptake and slow recovery.")
    return " ".join(m) if m else "Current environmental inputs indicate low immediate weather risk."
