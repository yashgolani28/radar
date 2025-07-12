import psycopg2
import psycopg2.extras
import subprocess
from datetime import datetime, timedelta
import logging

LOG_PATH = "/home/pi/radar/logs/activity_monitor.log"
INACTIVITY_THRESHOLD_MINUTES = 10

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s %(message)s')


def check_last_activity():
    try:
        conn = psycopg2.connect(
            dbname="radar_db",
            user="radar_user",
            password="securepass123",
            host="localhost"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM radar_data")
        last_ts = cursor.fetchone()[0]
        conn.close()

        if not last_ts:
            logging.warning("No activity in DB yet.")
            return True

        last_time = last_ts if isinstance(last_ts, datetime) else datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()

        if now - last_time > timedelta(minutes=INACTIVITY_THRESHOLD_MINUTES):
            logging.warning(f"No activity for over {INACTIVITY_THRESHOLD_MINUTES} minutes. Restarting services...")
            return True
        else:
            logging.info(f"Activity OK. Last seen at {last_time}")
            return False

    except Exception as e:
        logging.error(f"Error checking activity: {e}")
        return False


def restart_services():
    subprocess.call(["sudo", "systemctl", "restart", "radar_main.service"])
    subprocess.call(["sudo", "systemctl", "restart", "radar_flask.service"])
    logging.info("Services restarted.")


if __name__ == "__main__":
    if check_last_activity():
        restart_services()
