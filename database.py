import sqlite3
from datetime import datetime

DB_FILE = "radar_data.db"

def create_table():
    """Create radar_logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS radar_logs (
            timestamp REAL,
            datetime TEXT,
            sensor TEXT,
            object_id TEXT,
            type TEXT,
            confidence REAL,
            speed_kmh REAL,
            velocity REAL,
            distance REAL,
            direction TEXT,
            signal_level REAL,
            doppler_frequency REAL,
            snapshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(data):
    """Insert a single radar data record."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO radar_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        float(data['timestamp']),
        data['datetime'] if isinstance(data['datetime'], str) else data['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
        data.get('sensor', 'OPS243C'),
        data.get('object_id', ''),
        data.get('type', ''),
        float(data.get('confidence', 1.0)),
        float(data.get('speed_kmh', 0.0)),
        float(data.get('velocity', 0.0)),
        float(data.get('distance', 0.0)),
        data.get('direction', 'stationary'),
        float(data.get('signal_level', 0.0)),  # ✅ Force float
        float(data.get('doppler_frequency', 0.0))  # ✅ Force float
    ))
    conn.commit()
    conn.close()

def clear_database():
    """Delete all entries from the radar_logs table."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM radar_logs')
    conn.commit()
    conn.close()
