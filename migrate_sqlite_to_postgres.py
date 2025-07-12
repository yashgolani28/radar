import sqlite3
import psycopg2
import psycopg2.extras

# 1. Configuration
sqlite_path = "radar_data.db"
pg_config = {
    "dbname": "radar_db",
    "user": "radar_user",
    "password": "securepass123",
    "host": "localhost"
}

tables = ["radar_data", "users", "user_activity"]
schema = {
    "radar_data": [
        "timestamp", "datetime", "sensor", "object_id", "type", "confidence",
        "speed_kmh", "velocity", "distance", "direction", "signal_level",
        "doppler_frequency", "snapshot_path", "reviewed", "flagged",
        "radar_distance", "visual_distance"
    ],
    "users": ["username", "password_hash", "role", "created_at"],
    "user_activity": ["user_id", "last_activity"]
}

# 2. Migration Logic
try:
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    pg_conn = psycopg2.connect(**pg_config)
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute("SET search_path TO public;")

    for table in tables:
        cols = schema[table]
        col_str = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))

        sqlite_cursor.execute(f"SELECT {col_str} FROM {table}")
        rows = sqlite_cursor.fetchall()

        print(f"Migrating {len(rows)} rows from {table}...")

        for row in rows:
            values = [row[col] for col in cols]
            pg_cursor.execute(f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})", values)

    pg_conn.commit()
    print("✅ Migration complete.")

except Exception as e:
    print(f"❌ Migration failed: {e}")

finally:
    if 'sqlite_conn' in locals(): sqlite_conn.close()
    if 'pg_conn' in locals(): pg_conn.close()
