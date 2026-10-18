def severity(problem,confidence,humidity,temp,soil):
    if problem=="Healthy Plant": return "Low"
    if problem=="__UNCERTAIN__": return "Unknown"

    if problem=="Overwatering Stress":
        if soil>=90 or (soil>=82 and humidity>=85): return "High"
        return "Medium" if soil>=75 else "Low"

    if problem=="Nutrient Deficiency":
        return "High" if confidence>=90 else "Medium"

    if problem=="General Plant Stress":
        extreme = (temp>=38 or temp<=5 or soil<=10)
        return "High" if extreme else "Medium"

    score=0
    if confidence>=90: score+=2
    elif confidence>=75: score+=1
    if humidity>=80: score+=1
    if temp>=30 or temp<=10: score+=1
    if soil>=80 or soil<=20: score+=1
    return "High" if score>=4 else "Medium" if score>=2 else "Low"
