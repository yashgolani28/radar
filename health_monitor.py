#!/usr/bin/env python3

import psutil
import subprocess
import time
import logging
import os

# Optional: setup logging
log_file = "/home/pi/radar/logs/health_monitor.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Define safe thresholds
CPU_THRESHOLD = 85  # percent
MEM_THRESHOLD = 90  # percent
TEMP_THRESHOLD = 75.0  # Celsius

def get_temperature():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp = float(out.split("=")[1].split("'")[0])
        return temp
    except Exception as e:
        logging.warning(f"Could not read temperature: {e}")
        return 0.0

def restart_services(reason):
    logging.warning(f"[RESTART] Reason: {reason}")
    subprocess.call(["sudo", "systemctl", "restart", "radar_main.service"])
    subprocess.call(["sudo", "systemctl", "restart", "radar_flask.service"])

if __name__ == "__main__":
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    temp = get_temperature()

    logging.info(f"CPU: {cpu:.1f}%, Mem: {mem:.1f}%, Temp: {temp:.1f}°C")

    if cpu > CPU_THRESHOLD:
        restart_services(f"High CPU usage: {cpu:.1f}%")

    elif mem > MEM_THRESHOLD:
        restart_services(f"High memory usage: {mem:.1f}%")

    elif temp > TEMP_THRESHOLD:
        restart_services(f"High temperature: {temp:.1f}°C")
