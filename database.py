import sqlite3

conn = sqlite3.connect("radar_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS radar_data (
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
""")

conn.commit()
conn.close()
print("Table 'radar_data' created.")
