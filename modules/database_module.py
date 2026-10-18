
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "greenguardian_ai.db"

def connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con

def init_db():
    con = connect()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            scan_date TEXT NOT NULL,
            plant_type TEXT NOT NULL,
            image_source TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            soil_moisture REAL NOT NULL,
            last_watered_days INTEGER NOT NULL,
            detected_problem TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL NOT NULL,
            symptoms TEXT NOT NULL,
            treatment TEXT NOT NULL,
            medicine_name TEXT NOT NULL,
            medicine_cost TEXT NOT NULL,
            medicine_image TEXT NOT NULL,
            watering_advice TEXT NOT NULL,
            sunlight_advice TEXT NOT NULL,
            smart_watering_status TEXT NOT NULL,
            smart_watering_reason TEXT NOT NULL,
            weather_alert TEXT NOT NULL,
            detection_method TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    con.commit()
    con.close()

def save_scan(user_id, r):
    con = connect()
    con.execute("""
        INSERT INTO scans(
            user_id,scan_date,plant_type,image_source,temperature,humidity,soil_moisture,
            last_watered_days,detected_problem,severity,confidence,symptoms,treatment,
            medicine_name,medicine_cost,medicine_image,watering_advice,sunlight_advice,
            smart_watering_status,smart_watering_reason,weather_alert,detection_method
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        user_id,r["scan_date"],r["plant_type"],r["image_source"],r["temperature"],r["humidity"],
        r["soil_moisture"],r["last_watered_days"],r["detected_problem"],r["severity"],r["confidence"],
        r["symptoms"],r["treatment"],r["medicine_name"],r["medicine_cost"],r["medicine_image"],
        r["watering_advice"],r["sunlight_advice"],r["smart_watering_status"],r["smart_watering_reason"],
        r["weather_alert"],r["detection_method"]
    ))
    con.commit()
    con.close()

def get_scans(user_id):
    con=connect()
    rows=con.execute("""
        SELECT id,scan_date,plant_type,detected_problem,severity,confidence,
               smart_watering_status,medicine_name,medicine_cost,detection_method
        FROM scans WHERE user_id=? ORDER BY id DESC
    """,(user_id,)).fetchall()
    con.close()
    return rows

def stats(user_id):
    con=connect()
    total=con.execute("SELECT COUNT(*) FROM scans WHERE user_id=?",(user_id,)).fetchone()[0]
    healthy=con.execute("SELECT COUNT(*) FROM scans WHERE user_id=? AND detected_problem='Healthy Plant'",(user_id,)).fetchone()[0]
    high=con.execute("SELECT COUNT(*) FROM scans WHERE user_id=? AND severity='High'",(user_id,)).fetchone()[0]
    latest=con.execute("SELECT detected_problem FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 1",(user_id,)).fetchone()
    con.close()
    return total,healthy,high,(latest[0] if latest else "No scans")
