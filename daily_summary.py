from datetime import datetime
import os
import shutil
import psycopg2

# Paths
snapshot_dir = "/home/pi/radar_dashboard/snapshots"
log_output = []

# Timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_output.append(f"=== Daily Summary ({timestamp}) ===")

# Snapshot count
try:
    snapshot_files = [
        f for f in os.listdir(snapshot_dir)
        if os.path.isfile(os.path.join(snapshot_dir, f))
    ]
    snapshot_count = len(snapshot_files)
    log_output.append(f"Snapshot count: {snapshot_count}")
except Exception as e:
    log_output.append(f"Error counting snapshots: {e}")

# Disk space
try:
    total, used, free = shutil.disk_usage("/")
    free_gb = round(free / (1024**3), 2)
    log_output.append(f"Free disk space: {free_gb} GB")
except Exception as e:
    log_output.append(f"Error getting disk usage: {e}")

# DB record count + PostgreSQL size
try:
    conn = psycopg2.connect(
        dbname="radar_db",
        user="radar_user",
        password="securepass123",
        host="localhost"
    )
    cursor = conn.cursor()
    
    # Row count
    cursor.execute("SELECT COUNT(*) FROM radar_data")
    count = cursor.fetchone()[0]
    log_output.append(f"Radar DB records: {count}")

    # Optional: PostgreSQL DB size
    cursor.execute("SELECT pg_size_pretty(pg_database_size('radar_db'))")
    db_size = cursor.fetchone()[0]
    log_output.append(f"PostgreSQL DB size: {db_size}")

    conn.close()
except Exception as e:
    log_output.append(f"Error accessing PostgreSQL: {e}")

# Final output
summary = "\n".join(log_output)
print(summary)
