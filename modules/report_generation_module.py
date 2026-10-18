
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.units import mm

def make_pdf(user_name,r):
    b=BytesIO()
    doc=SimpleDocTemplate(b,pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=15*mm,bottomMargin=15*mm)
    s=getSampleStyleSheet()
    story=[Paragraph("GreenGuardian AI",s["Title"]),
           Paragraph("Smart Plant Disease Detection and Care Advisory System",s["Heading2"]),Spacer(1,8)]
    rows=[
      ["Registered User",user_name],["Report Date",r["scan_date"]],["Plant",r["plant_type"]],
      ["Detection Method",r["detection_method"]],["Detected Problem",r["detected_problem"]],
      ["Severity",r["severity"]],["Confidence",f'{r["confidence"]}%'],
      ["Temperature",f'{r["temperature"]} °C'],["Humidity",f'{r["humidity"]}%'],
      ["Soil Moisture",f'{r["soil_moisture"]}%'],["Symptoms",r["symptoms"]],
      ["Treatment",r["treatment"]],["Medicine",r["medicine_name"]],["Approx. Cost",r["medicine_cost"]],
      ["Watering Advice",r["watering_advice"]],["Smart Watering",r["smart_watering_status"]],
      ["Reason",r["smart_watering_reason"]],["Sunlight",r["sunlight_advice"]],
      ["Weather Risk",r["weather_alert"]]]
    t=Table(rows,colWidths=[43*mm,127*mm])
    t.setStyle(TableStyle([
      ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#E7F7ED")),
      ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CFE3D7")),
      ("VALIGN",(0,0),(-1,-1),"TOP"),
      ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
      ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)
    ]))
    story += [t,Spacer(1,8),Paragraph(
      "Prototype advisory note: GreenGuardian AI supports gardening decisions and is not a replacement for professional horticultural diagnosis. Medicine images are illustrative and costs are approximate.",
      s["Italic"])]
    doc.build(story); b.seek(0); return b
