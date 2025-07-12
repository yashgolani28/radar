import os
import smtplib
from email.message import EmailMessage
from app import get_db_connection, load_config
from report import generate_pdf_report
from datetime import datetime
import requests

# 1. Generate the report
filename = f"radar_daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
filepath = os.path.join("backups", filename)
os.makedirs("backups", exist_ok=True)

with get_db_connection() as conn:
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("""
        SELECT datetime, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
            radar_distance, visual_distance, direction, signal_level, doppler_frequency,
            reviewed, flagged, snapshot_path
        FROM radar_data
        WHERE DATE(datetime) = CURRENT_DATE
        ORDER BY datetime ASC
    """)
    rows = cursor.fetchall()

data = [dict(row) for row in rows]

from collections import Counter
speeds = [float(d["speed_kmh"]) for d in data if d.get("speed_kmh")]
types = [d["type"].upper() for d in data if d.get("type")]
directions = [d["direction"].lower() for d in data if d.get("direction")]

summary = {
    "total_records": len(data),
    "avg_speed": sum(speeds) / len(speeds) if speeds else 0.0,
    "top_speed": max(speeds) if speeds else 0.0,
    "lowest_speed": min(speeds) if speeds else 0.0,
    "most_detected_object": Counter(types).most_common(1)[0][0] if types else "N/A",
    "approaching_count": sum(1 for d in directions if d == "approaching"),
    "stationary_count": sum(1 for d in directions if d == "stationary"),
    "departing_count": sum(1 for d in directions if d == "departing"),
    "last_detection": data[-1].get("datetime") if data else "N/A",
    "speed_limits": load_config().get("dynamic_speed_limits", {})
}

charts = {}
try:
    resp = requests.get("http://127.0.0.1:5000/api/charts?days=0")
    if resp.ok:
        raw_charts = resp.json()
        charts = {
            title: chart
            for title, chart in raw_charts.items()
            if chart.get("labels") and chart.get("data") and any(v > 0 for v in chart["data"])
        }
except Exception as e:
    print(f"[CHART FETCH FAIL] {e}")

generate_pdf_report(filepath, data=data, summary=summary, logo_path="/home/pi/radar/static/essi_logo.jpeg", charts=charts)

# 2. Email it
EMAIL_FROM = "yashgolani.essi@gmail.com"
EMAIL_TO = "yashgolani.essi@gmail.com"
EMAIL_PASS = "pwnn bopj tohy buyd"  

msg = EmailMessage()
msg["Subject"] = f"[ESSI] Daily Radar Detection Report – {datetime.now().strftime('%B %d, %Y')}"
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO
msg.set_content(
    f"""
Dear Team,

Please find attached the radar activity report for {datetime.now().strftime('%B %d, %Y')}. This report includes summary statistics and analytics for all detections recorded today.

If you have any questions or need further details, feel free to reach out.

Best regards,  
Radar Based Speed Detection System  
ELKOSTA SECURITY SYSTEMS INDIA (ESSI)
"""
)


with open(filepath, "rb") as f:
    msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=filename)

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(EMAIL_FROM, EMAIL_PASS)
    smtp.send_message(msg)
