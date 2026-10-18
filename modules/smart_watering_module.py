RULES={
"Tomato":(45,75,2),
"Apple":(30,65,4),
"Rose":(35,70,3),
"Grape":(30,65,4),
"Corn":(35,65,3)
}

def smart_watering(plant,problem,soil,last_days,humidity,temp):
    mn,mx,maxdays=RULES.get(plant,(35,70,3))
    reasons=[]

    # Proposal-specific overwatering stress must always block additional watering.
    if problem=="Overwatering Stress":
        status="Do Not Water"
        reasons.append("Overwatering stress is indicated; allow soil moisture to fall before watering again.")
        if soil>mx:
            reasons.append("Current soil moisture is above the preferred range.")
        if humidity>=75:
            reasons.append("High humidity can slow soil drying and increase disease risk.")
        return status," ".join(reasons),f"{mn}% - {mx}%"

    if soil>mx:
        status="Do Not Water"; reasons.append("Soil moisture is already high.")
    elif soil<mn:
        status="Water Now"; reasons.append("Soil moisture is below the preferred range.")
    elif last_days>=maxdays:
        status="Light Watering Recommended"; reasons.append("The normal watering interval has been reached.")
    else:
        status="No Water Needed Today"; reasons.append("Soil moisture is within the preferred range.")

    # Never overturn a 'Do Not Water' decision just because the day is hot.
    if temp>=30 and status not in {"Do Not Water"}:
        status="Water Now"; reasons.append("High temperature may increase water loss.")

    if problem=="General Plant Stress" and soil<=20:
        reasons.append("Dry-soil stress is contributing to the general stress assessment.")

    if humidity>=75 and any(k in problem.lower() for k in ["blight","mold","mildew","rot","rust","scab","spot"]):
        reasons.append("Keep water at soil level because humid conditions can increase fungal risk.")
    return status," ".join(reasons),f"{mn}% - {mx}%"
