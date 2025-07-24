from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, session, jsonify, flash, abort, Response, stream_with_context, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from requests.auth import HTTPDigestAuth
from report import generate_pdf_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import requests
import psycopg2
import psycopg2.extras
import csv
import os
import json
import traceback
import shutil
import cv2
import threading
import numpy as np
import psutil
import subprocess
import zipfile
import time
import uuid
from bounding_box import annotate_speeding_object
from config_utils import load_config
from camera import capture_snapshot
from check_radar_connection import check_radar_connection
from collections import Counter, deque
from datetime import datetime, timedelta
from io import BytesIO
import logging
from flask_socketio import SocketIO
import base64
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Cam Setup
camera_capture = None
last_frame = None
camera_lock = threading.Lock()
camera_frame = None
config = load_config()
selected = config.get("selected_camera", 0)
cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
camera_url = cam.get("url")
camera_auth = HTTPDigestAuth(cam.get("username"), cam.get("password")) if cam.get("username") else None

# Configuration
SNAPSHOT_FOLDER = "snapshots"
BACKUP_FOLDER = "backups"
CONFIG_FILE = "app_config.json"

# --- User Management ---
class User(UserMixin):
    def __init__(self, id_, username, password_hash, role="viewer"):
        self.id = id_
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def get_id(self):
        return str(self.id) 
    @property
    def is_authenticated(self):
        return True  
    
def get_user_by_id(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        return User(*row) if row else None

def get_user_by_username(username):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        return User(*row) if row else None
    
def save_cameras_to_db(cameras, selected_idx):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras")
        for i, cam in enumerate(cameras):
            cursor.execute("""
                INSERT INTO cameras (url, username, password, is_active, stream_type)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                cam.get("url"),
                cam.get("username"),
                cam.get("password"),
                i == selected_idx,
                cam.get("stream_type", "mjpeg")
            ))
        conn.commit()

def load_cameras_from_db():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT url, username, password, is_active, stream_type FROM cameras ORDER BY id")
        rows = cursor.fetchall()
        cameras = []
        selected = 0
        for i, row in enumerate(rows):
            cameras.append({
                "url": row["url"],
                "username": row["username"],
                "password": row["password"],
                "stream_type": row["stream_type"] or "mjpeg"
            })
            if row["is_active"]:
                selected = i
        return cameras, selected

@contextmanager
def get_db_connection():
    conn = None
    max_retries = 3
    retry_delay = 0.1

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname="radar_db",
                user="radar_user",
                password="securepass123",
                host="localhost"
            )
            break
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"PostgreSQL connection failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected PostgreSQL error: {e}")
            raise

    try:
        yield conn
    finally:
        if conn:
            conn.close()

def update_user_activity(user_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                INSERT INTO user_activity (user_id, last_activity)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET last_activity = EXCLUDED.last_activity
            """, (user_id, datetime.now().isoformat()))
            conn.commit()
    except Exception as e:
        logger.error(f"Error updating user activity: {e}")

def get_active_users(minutes=30):
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT u.id, u.username, u.role, ua.last_activity
                FROM users u
                JOIN user_activity ua ON u.id = ua.user_id
                WHERE CAST(ua.last_activity AS TIMESTAMP) >= %s
                ORDER BY CAST(ua.last_activity AS TIMESTAMP) DESC
            """, (cutoff,))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
        return []

def clean_inactive_sessions():
    try:
        cutoff = datetime.now() - timedelta(hours=24)
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("DELETE FROM user_activity WHERE datetime(last_activity) < %s", (cutoff.isoformat(),))
            conn.commit()
    except Exception as e:
        logger.error(f"Error cleaning inactive sessions: {e}")

def is_admin():
    return current_user.is_authenticated and getattr(current_user, "role", None) == "admin"

def load_config():
    """Load application configuration"""
    default_config = {
        "cooldown_seconds": 0.5,
        "retention_days": 30,
        "selected_camera": 0,
        "annotation_conf_threshold": 0.5,
        "label_format": "{type} | {speed:.1f} km/h",
        "cameras": ["Camera 1", "Camera 2", "Camera 3"],
        "dynamic_speed_limits": {
            "default": 3.0,
            "HUMAN": 4.0,
            "CAR": 60.0,
            "TRUCK": 50.0,
            "BIKE": 20.0,
            "UNKNOWN": 25.0
        }
    }

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

            # Fill in missing default keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value

            # Ensure dynamic_speed_limits has a default
            if "dynamic_speed_limits" not in config or not isinstance(config["dynamic_speed_limits"], dict):
                config["dynamic_speed_limits"] = default_config["dynamic_speed_limits"]

            if "default" not in config["dynamic_speed_limits"]:
                config["dynamic_speed_limits"]["default"] = default_config["dynamic_speed_limits"]["default"]

            return config

    except (FileNotFoundError, json.JSONDecodeError):
        return default_config


def save_config(config):
    """Save application configuration"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False

def ensure_directories():
    """Create necessary directories if they don't exist"""
    for directory in [SNAPSHOT_FOLDER, BACKUP_FOLDER]:
        os.makedirs(directory, exist_ok=True)

def save_model_metadata(accuracy, method):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO model_info (accuracy, method, updated_at) VALUES (%s, %s, %s)",
                           (accuracy, method, datetime.now()))
            conn.commit()
    except Exception as e:
        logger.error(f"[MODEL INFO SAVE ERROR] {e}")

def get_model_metadata():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM model_info ORDER BY updated_at DESC LIMIT 2")
            rows = cursor.fetchall()
            if not rows:
                return None
            latest = rows[0]
            prev = rows[1] if len(rows) > 1 else None
            change = (latest['accuracy'] - prev['accuracy']) if prev else None
            return {
                "accuracy": latest['accuracy'],
                "updated_at": latest['updated_at'].strftime("%Y-%m-%d %H:%M:%S"),
                "method": latest['method'],
                "change": round(change, 2) if change is not None else None
            }
    except Exception as e:
        logger.error(f"[MODEL INFO LOAD ERROR] {e}")
        return None

def validate_snapshots():
    """Validate snapshot paths in database and clean up missing files"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT id, snapshot_path FROM radar_data WHERE snapshot_path IS NOT NULL")
            rows = cursor.fetchall()

            invalid_count = 0
            for row in rows:
                path = row['snapshot_path']
                if not os.path.exists(path):
                    cursor.execute("UPDATE radar_data SET snapshot_path = NULL WHERE id = %s", (row['id'],))
                    invalid_count += 1
            conn.commit()
            logger.info(f"Validated snapshots: {invalid_count} invalid paths cleaned")
            return invalid_count
    except Exception as e:
        logger.error(f"Error validating snapshots: {e}")
        return 0

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
    app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production')
    app.permanent_session_lifetime = timedelta(minutes=30)
    
    # Login manager setup
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)
    
    @app.context_processor
    def inject_globals():
        return {"now": datetime.now(), "is_admin": is_admin()}
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors.html', message="Page not found", error_code=404), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors.html', message="Internal server error", error_code=500), 500
    
    @app.errorhandler(413)
    def file_too_large(error):
        return render_template('errors.html', message="File too large", error_code=413), 413
    
    @app.errorhandler(Exception)
    def database_error(error):
        logger.error(f"Database error: {error}")
        logger.error(f"Unhandled error: {error}\n{traceback.format_exc()}")
        return render_template('errors.html', message="Database connection error", error_code=500), 500
    
    @app.route("/camera_feed")
    @login_required
    def camera_feed():
        def generate():
            try:
                config = load_config()
                cameras, selected = load_cameras_from_db()
                cam = cameras[selected] if cameras and selected < len(cameras) else {}

                url = cam.get("url")
                username = cam.get("username")
                password = cam.get("password")
                stream_type = cam.get("stream_type", "mjpeg").lower()
                auth = HTTPDigestAuth(username, password) if username and password else None
                auth_str = f"{username}:{password}@" if username and password else ""

                if not url:
                    return

                # RTSP stream (MJPEG)
                if stream_type == "rtsp":
                    auth_str = f"{username}:{password}@" if username and password else ""
                    if url.startswith("rtsp://") and "@" not in url and auth_str:
                        rtsp_url = url.replace("rtsp://", f"rtsp://{auth_str}")
                    else:
                        rtsp_url = url

                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-rtsp_transport", "tcp",
                        "-user_agent", "Mozilla/5.0",
                        "-i", rtsp_url,
                        "-f", "mjpeg",
                        "-qscale:v", "2",
                        "-r", "5",
                        "-"
                    ]
                    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    buffer = b""
                    while True:
                        chunk = process.stdout.read(4096)
                        if not chunk:
                            break
                        buffer += chunk
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9')
                        if start != -1 and end != -1 and end > start:
                            frame = buffer[start:end+2]
                            buffer = buffer[end+2:]
                            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    process.kill()

                # Snapshot fallback
                elif stream_type == "snapshot":
                    while True:
                        try:
                            response = requests.get(url, auth=auth, timeout=5)
                            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + response.content + b"\r\n")
                            else:
                                logger.warning(f"[CAMERA_FEED] Snapshot failed, status={response.status_code}")
                        except Exception as e:
                            logger.error(f"[CAMERA_FEED] Snapshot error: {e}")
                        time.sleep(0.5)

                # MJPEG stream proxy
                else:
                    with requests.get(url, auth=auth, stream=True, timeout=10) as r:
                        if r.status_code != 200:
                            logger.error(f"[CAMERA_FEED] MJPEG stream returned status {r.status_code}")
                            return
                        logger.info("[CAMERA_FEED] Connected to MJPEG stream.")
                        buffer = b""
                        for chunk in r.iter_content(chunk_size=4096):
                            if not chunk:
                                continue
                            buffer += chunk
                            start = buffer.find(b'\xff\xd8')
                            end = buffer.find(b'\xff\xd9')
                            if start != -1 and end != -1 and end > start:
                                frame = buffer[start:end+2]
                                buffer = buffer[end+2:]
                                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            except Exception as e:
                logger.exception(f"[CAMERA_FEED] Fatal error: {e}")
                time.sleep(2)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route("/api/reload_config", methods=["POST"])
    @login_required
    def reload_config():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            # Write to a file flag or message queue
            with open("reload_flag.txt", "w") as f:
                f.write(str(time.time()))
            return jsonify({"status": "ok", "message": "Config reload requested."})
        except Exception as e:
            logger.error(f"[RELOAD API ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500

    @app.route("/manual_snapshot", methods=["POST"])
    @login_required
    def manual_snapshot():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            now = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            config = load_config()
            selected = config.get("selected_camera", 0)
            cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}

            snapshot_url = cam.get("url")
            username = cam.get("username")
            password = cam.get("password")
            auth = HTTPDigestAuth(username, password) if username and password else None
            response = requests.get(snapshot_url, auth=auth, timeout=5)

            if response.status_code != 200 or not response.content:
                return jsonify({"error": "Snapshot capture failed"}), 500

            snapshot_path = os.path.join(SNAPSHOT_FOLDER, f"manual_{timestamp}.jpg")
            with open(snapshot_path, "wb") as f:
                f.write(response.content)

            label = "MANUAL SNAPSHOT"
            conf_thresh = load_config().get("annotation_conf_threshold", 0.5)
            annotated_path, visual_distance, radar_distance = annotate_speeding_object(
                image_path=snapshot_path,
                radar_distance=0.0,
                label=label,
                min_confidence=conf_thresh
            )

            if not annotated_path:
                return jsonify({"error": "Annotation failed"}), 500

            # Insert into DB
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    INSERT INTO radar_data (
                        timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                        velocity, distance, radar_distance, visual_distance,
                        direction, signal_level, doppler_frequency, snapshot_path
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    now,
                    datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                    "Manual",
                    f"manual_{uuid.uuid4().hex[:6]}",
                    "UNKNOWN",
                    0.0,
                    0.0,
                    0.0,
                    radar_distance,
                    radar_distance,
                    visual_distance,
                    "manual",
                    0.0,
                    0.0,
                    annotated_path
                ))
                conn.commit()

            return jsonify({"status": "ok", "message": "Snapshot captured successfully."})

        except Exception as e:
            logger.error(f"[MANUAL SNAPSHOT ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500
        
    @app.route("/logs")
    @login_required
    def view_logs():
        try:
            # Get recent logs
            log_path = os.path.join("logs", "radar.log")
            logs = []
            if os.path.isfile(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    logs = f.readlines()[-1000:]
            else:
                logs = ["[INFO] Log file not found: logs/radar.log"]

            return render_template("logs.html", logs=logs)
        
        except Exception as e:
            import traceback
            err_msg = f"Exception in /logs: {e}\n{traceback.format_exc()}"
            return f"<pre style='color:red;'>{err_msg}</pre>", 500

    @app.route("/api/logs")
    @login_required
    def api_logs():
        try:
            log_path = os.path.join("logs", "radar.log")
            offset = int(request.args.get("offset", 0))
            limit = int(request.args.get("limit", 100))
            max_lines = offset + limit

            if not os.path.exists(log_path):
                return jsonify({"logs": [], "has_more": False})

            # Use deque to load only necessary tail
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = deque(f, maxlen=max_lines)

            logs = list(all_lines)
            paginated = logs[-limit:]

            return jsonify({
                "logs": [line.strip() for line in paginated],
                "has_more": len(all_lines) >= max_lines
            })

        except Exception as e:
            logger.exception(f"[LOGS API ERROR] {e}")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"].strip()
            password = request.form["password"]
            user = get_user_by_username(username)
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                return redirect(url_for("index"))
            return render_template("login.html", error="Invalid credentials")
        return render_template("login.html")
    
    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out successfully", "success")
        return redirect(url_for("login"))
    
    @app.route("/")
    @login_required
    def index():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                # Get recent detections
                cursor.execute("""
                    SELECT datetime, type, speed_kmh, distance, radar_distance, visual_distance, direction, snapshot_path, object_id, confidence
                    FROM radar_data 
                    WHERE snapshot_path IS NOT NULL
                    ORDER BY datetime DESC
                    LIMIT 10
                """)
                rows = cursor.fetchall()
    
                # Get summary statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN LOWER(COALESCE(type, '')) LIKE '%human%' OR LOWER(COALESCE(type, '')) LIKE '%person%' THEN 1 ELSE 0 END) as humans,
                        SUM(CASE WHEN LOWER(COALESCE(type, '')) LIKE '%vehicle%' OR LOWER(COALESCE(type, '')) LIKE '%car%' THEN 1 ELSE 0 END) as vehicles,
                        AVG(CASE WHEN speed_kmh IS NOT NULL AND speed_kmh >= 0 THEN speed_kmh END) as avg_speed,
                        MAX(datetime) as last_detection
                    FROM radar_data
                """)
                stats = cursor.fetchone()
                
                total, humans, vehicles, avg_speed, last_detection = (
                    stats['total'] if stats else 0,
                    stats['humans'] if stats else 0,
                    stats['vehicles'] if stats else 0,
                    stats['avg_speed'] if stats else 0,
                    stats['last_detection'] if stats else None
                )
    
                # Process snapshots data
                snapshots = []
                for r in rows:
                    snapshot_data = {
                        "datetime": r['datetime'] or "N/A",
                        "type": r['type'] or "UNKNOWN",
                        "speed": round(float(r['speed_kmh']) if r['speed_kmh'] is not None else 0, 2),
                        "distance": round(float(r['distance']) if r['distance'] is not None else 0, 2),
                        "radar_distance": round(float(r['radar_distance']) if r['radar_distance'] is not None else 0, 2),
                        "visual_distance": round(float(r['visual_distance']) if r['visual_distance'] is not None else 0, 2),
                        "direction": r['direction'] or "N/A",
                        "image": os.path.basename(r['snapshot_path']) if r['snapshot_path'] else None,
                        "object_id": r['object_id'] or "N/A",
                        "confidence": round(float(r['confidence']) if r['confidence'] is not None else 0, 2),
                    }
                    try:
                        label = f"{snapshot_data['type']} | {snapshot_data['speed']} km/h | {snapshot_data['distance']} m | {snapshot_data['direction']}"
                    except Exception as e:
                        logger.warning(f"Snapshot label formatting failed: {e}")
                        label = "UNKNOWN"
                    snapshot_data["label"] = label
                    snapshots.append(snapshot_data)

                log_path = os.path.join("logs", "radar.log")
                logs = []
                if os.path.isfile(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        logs = f.readlines()[-15:]
                else:
                    logs = ["[INFO] Log file not found: logs/radar.log"]

                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        temp_c = int(f.read()) / 1000.0
                except Exception:
                    temp_c = None

                summary = {
                    "total": total or 0,
                    "humans": humans or 0,
                    "vehicles": vehicles or 0,
                    "average_speed": round(avg_speed, 2) if avg_speed else 0,
                    "last_detection": (
                        last_detection.strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(last_detection, datetime) else str(last_detection or "N/A")
                    ),
                    "logs": logs,
                    "pi_temperature": round(temp_c, 1) if temp_c is not None else "N/A",
                    "cpu_load": round(os.getloadavg()[0], 2)  
                }
    
                return render_template("index.html", snapshots=snapshots, summary=summary, config=load_config())
                
        except Exception as e:
            logger.error(f"Error in index route: {e}")
            flash("Error loading dashboard data", "error")
            summary = {
                "total": 0,
                "humans": 0,
                "vehicles": 0,
                "average_speed": 0,
                "last_detection": "N/A",
                "logs": [],
                "pi_temperature": 0.0,
                "cpu_load": 0.0
            }
            return render_template("index.html", snapshots=[], summary=summary, config=load_config())
        
    @app.route("/api/status")
    @login_required
    def api_status():
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

    @app.route("/api/charts")
    def api_charts():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                try:
                    days = int(request.args.get("days", 30))
                except ValueError:
                    days = 30

                # --- Speed histogram ---
                if days <= 0:
                    cursor.execute("""
                        SELECT speed_kmh 
                        FROM radar_data 
                        WHERE speed_kmh IS NOT NULL 
                        AND speed_kmh >= 0 AND speed_kmh <= 200
                        AND DATE(datetime::TIMESTAMP) = CURRENT_DATE
                    """)
                else:
                    cursor.execute("""
                        SELECT speed_kmh 
                        FROM radar_data 
                        WHERE speed_kmh IS NOT NULL 
                        AND speed_kmh >= 0 AND speed_kmh <= 200
                        AND CAST(datetime AS TIMESTAMP) >= CURRENT_DATE - INTERVAL %s DAY
                    """, (str(days),))

                speed_rows = cursor.fetchall()
                speeds = [float(row["speed_kmh"]) for row in speed_rows if row["speed_kmh"] is not None]

                speed_bins = list(range(0, 101, 10))
                speed_labels = [f"{i}-{i+9}" for i in speed_bins[:-1]] + ["100+"]
                speed_counts = [0] * len(speed_labels)

                for speed in speeds:
                    if speed >= 100:
                        speed_counts[-1] += 1
                    else:
                        index = int(speed // 10)
                        speed_counts[index] += 1

                # --- Direction breakdown ---
                if days <= 0:
                    cursor.execute("""
                        SELECT direction 
                        FROM radar_data 
                        WHERE direction IS NOT NULL AND TRIM(direction) != ''
                        AND DATE(datetime::TIMESTAMP) = CURRENT_DATE
                    """)
                else:
                    cursor.execute("""
                        SELECT direction 
                        FROM radar_data 
                        WHERE direction IS NOT NULL AND TRIM(direction) != ''
                        AND CAST(datetime AS TIMESTAMP) >= CURRENT_DATE - INTERVAL %s DAY
                    """, (str(days),))

                direction_rows = cursor.fetchall()
                directions = [row["direction"].strip().lower() for row in direction_rows if row["direction"]]

                direction_mapping = {'approaching': 'approaching', 'stationary': 'stationary', 'departing': 'departing'}
                normalized = [direction_mapping.get(d, 'other') for d in directions]
                direction_labels = ['approaching', 'stationary', 'departing', 'other']
                direction_data = [normalized.count(label) for label in direction_labels]

                # --- Violations per hour ---
                if days <= 0:
                    cursor.execute("""
                        SELECT TO_CHAR(datetime::TIMESTAMP, 'HH24') as hour, COUNT(*) as count
                        FROM radar_data 
                        WHERE speed_kmh > 0 AND DATE(datetime::TIMESTAMP) = CURRENT_DATE
                        GROUP BY hour
                        ORDER BY hour
                    """)
                else:
                    cursor.execute("""
                        SELECT TO_CHAR(datetime::TIMESTAMP, 'HH24') as hour, COUNT(*) as count
                        FROM radar_data 
                        WHERE speed_kmh > 0 AND CAST(datetime AS TIMESTAMP) >= CURRENT_DATE - INTERVAL %s DAY
                        GROUP BY hour
                        ORDER BY hour
                    """, (str(days),))

                hourly_rows = cursor.fetchall()
                hour_labels = [f"{int(r['hour']):02d}:00" for r in hourly_rows]
                hour_data = [r['count'] for r in hourly_rows]

                return jsonify({
                    "speed_histogram": {
                        "labels": speed_labels,
                        "data": speed_counts
                    },
                    "direction_breakdown": {
                        "labels": direction_labels,
                        "data": direction_data
                    },
                    "violations_per_hour": {
                        "labels": hour_labels,
                        "data": hour_data
                    }
                })

        except psycopg2.Error as e:
            logger.exception("Database error in chart API")
            return jsonify({"error": "Database error"}), 500
        except Exception as e:
            logger.exception("Unhandled error in chart API")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/gallery")
    @login_required
    def gallery():
        # Get filter parameters
        obj_type = request.args.get("type", "").upper()
        min_speed = float(request.args.get("min_speed") or 0)
        max_speed = float(request.args.get("max_speed") or 999)
        direction = request.args.get("direction", "").lower()
        object_id = request.args.get("object_id", "")
        start_date = request.args.get("start_date", "")
        end_date = request.args.get("end_date", "")
        min_confidence = float(request.args.get("min_confidence") or 0)
        max_confidence = float(request.args.get("max_confidence") or 1)
        reviewed_only = request.args.get("reviewed_only") == "1"
        flagged_only = request.args.get("flagged_only") == "1"
        unannotated_only = request.args.get("unannotated_only") == "1"
        download = request.args.get("download", "0") == "1"

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                # Build dynamic query
                query = """
                    SELECT datetime, type, speed_kmh, radar_distance, visual_distance, direction, snapshot_path, 
                        object_id, confidence, reviewed, flagged
                    FROM radar_data WHERE 1=1
                """
                params = []
                
                # Add filters
                if min_speed > 0 or max_speed < 999:
                    query += " AND speed_kmh BETWEEN %s AND %s"
                    params.extend([min_speed, max_speed])
                
                if obj_type:
                    query += " AND UPPER(COALESCE(type, '')) LIKE %s"
                    params.append(f"%{obj_type}%")
                
                if direction:
                    query += " AND LOWER(COALESCE(direction, '')) LIKE %s"
                    params.append(f"%{direction}%")
                
                if object_id:
                    query += " AND COALESCE(object_id, '') LIKE %s"
                    params.append(f"%{object_id}%")
                
                if start_date:
                    query += " AND DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) >= %s"
                    params.append(start_date)
                if end_date:
                    query += " AND DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) <= %s"
                    params.append(end_date)
                
                if min_confidence > 0 or max_confidence < 1:
                    query += " AND confidence BETWEEN %s AND %s"
                    params.extend([min_confidence, max_confidence])
                
                # Add annotation status filters
                if reviewed_only:
                    query += " AND reviewed = 1"
                
                if flagged_only:
                    query += " AND flagged = 1"
                
                if unannotated_only:
                    query += " AND reviewed = 0 AND flagged = 0"

                query += " ORDER BY COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp)) DESC LIMIT 1000"
                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Process results
                snapshots = []
                for r in rows:
                    snapshot_data = {
                        "filename": os.path.basename(r['snapshot_path']) if r['snapshot_path'] else "no_image.jpg",
                        "datetime": r['datetime'] or "N/A",
                        "type": r['type'] or "UNKNOWN",
                        "speed": round(float(r['speed_kmh']) if r['speed_kmh'] is not None else 0, 2),
                        "radar_distance": round(float(r['radar_distance']) if r['radar_distance'] is not None else 0, 2),
                        "visual_distance": round(float(r['visual_distance']) if r['visual_distance'] is not None else 0, 2),
                        "direction": r['direction'] or "N/A",
                        "object_id": r['object_id'] or "N/A",
                        "confidence": round(float(r['confidence']) if r['confidence'] is not None else 0, 2),
                        "reviewed": r['reviewed'] or 0,
                        "flagged": r['flagged'] or 0,
                        "path": r['snapshot_path'] if r['snapshot_path'] and os.path.exists(r['snapshot_path']) else None
                    }
                    snapshot_data["label"] = f"{snapshot_data['type']} | {snapshot_data['speed']} km/h | {snapshot_data['radar_distance']} m | {snapshot_data['direction']}"
                    snapshots.append(snapshot_data)

                # Handle download request
                if download and snapshots:
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, 'w') as zipf:
                        for snap in snapshots:
                            path = snap.get('path')
                            if path and os.path.isfile(path):
                                try:
                                    zipf.write(path, arcname=snap['filename'])
                                except Exception as e:
                                    logger.warning(f"Failed to add {snap['filename']} to zip: {e}")
                            else:
                                logger.warning(f"Missing or invalid path for snapshot: {snap['filename']}")
                                continue
                    if not zipf.namelist():
                        return jsonify({"error": "No valid files to download"}), 400
                    buffer.seek(0)
                    response = send_file(buffer, mimetype='application/zip', as_attachment=True,
                        download_name="filtered_snapshots.zip", conditional=True)
                    response.headers['Content-Length'] = buffer.getbuffer().nbytes  
                    return response

                return render_template("gallery.html", snapshots=snapshots)
                
        except Exception as e:
            logger.error(f"Error in gallery route: {e}")
            flash("Error loading gallery data", "error")
            return render_template("gallery.html", snapshots=[])
    
    @app.route("/mark_snapshot", methods=["POST"])
    @login_required
    def mark_snapshot():
        try:
            data = request.get_json()
            snapshot = data.get("snapshot")
            action = data.get("action")
    
            if not snapshot or action not in ("reviewed", "flagged"):
                return jsonify({"error": "Invalid input"}), 400
    
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute(f"SELECT {action} FROM radar_data WHERE snapshot_path LIKE %s", (f"%{snapshot}",))
                current = cursor.fetchone()
                new_value = 0 if current and current[action] == 1 else 1
    
                cursor.execute(f"UPDATE radar_data SET {action} = %s WHERE snapshot_path LIKE %s", 
                             (new_value, f"%{snapshot}",))
                conn.commit()
    
                return jsonify({"status": "updated", "new_value": new_value})
                
        except Exception as e:
            logger.error(f"Error marking snapshot: {e}")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/snapshots/<filename>")
    @login_required
    def serve_snapshot(filename):
        try:
            return send_from_directory(SNAPSHOT_FOLDER, filename)
        except FileNotFoundError:
            return "File not found", 404
    
    @app.route("/export_pdf")
    @login_required
    def export_pdf():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    SELECT datetime, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                        radar_distance, visual_distance, direction, signal_level, doppler_frequency,
                        reviewed, flagged, snapshot_path
                    FROM radar_data
                    ORDER BY datetime DESC
                    LIMIT 100
                """)
                rows = cursor.fetchall()

            data = [dict(row) for row in rows]

            speed_values = [float(d.get("speed_kmh") or 0) for d in data if d.get("speed_kmh") not in [None, ""]]
            avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0

            from collections import Counter

            speeds = [float(d["speed_kmh"]) for d in data if d.get("speed_kmh") is not None]
            types = [d["type"].upper() for d in data if d.get("type")]
            directions = [d["direction"].lower() for d in data if d.get("direction")]

            summary = {
                "total_records": len(data),
                "avg_speed": sum(speeds)/len(speeds) if speeds else 0.0,
                "top_speed": max(speeds) if speeds else 0.0,
                "lowest_speed": min(speeds) if speeds else 0.0,
                "most_detected_object": Counter(types).most_common(1)[0][0] if types else "N/A",
                "approaching_count": sum(1 for d in directions if d == "approaching"),
                "stationary_count": sum(1 for d in directions if d == "stationary"),
                "departing_count": sum(1 for d in directions if d == "departing"),
                "last_detection": data[0].get("datetime") if data else "N/A",
                "speed_limits": load_config().get("dynamic_speed_limits", {})
            }

            try:
                response = requests.get("http://127.0.0.1:5000/api/charts")
                response.raise_for_status()
                charts = response.json()
            except Exception as e:
                logger.error(f"[CHART FETCH ERROR] {e}")
                charts = {}

            logo_path = "/home/pi/radar/static/essi_logo.jpeg"
            filename = f"radar_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join("backups", filename)
            os.makedirs("backups", exist_ok=True)

            config = load_config()
            summary["speed_limits"] = config.get("dynamic_speed_limits", {})

            generate_pdf_report(filepath, data=data, summary=summary, logo_path=logo_path, charts=charts)
            return send_file(filepath, as_attachment=True)

        except Exception as e:
            logger.error(f"[EXPORT_PDF_ERROR] {e}")
            return str(e), 500

    @app.route("/export_filtered_pdf")
    @login_required
    def export_filtered_pdf():
        try:
            params = request.args.to_dict()
            filters = params.copy()

            query = """
                SELECT datetime, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                    radar_distance, visual_distance, direction, signal_level, doppler_frequency,
                    reviewed, flagged, snapshot_path
                FROM radar_data
                WHERE 1=1
            """
            sql_params = []

            if 'type' in params and params['type']:
                query += " AND UPPER(COALESCE(type, '')) LIKE %s"
                sql_params.append(f"%{params['type'].upper()}%")

            if 'min_speed' in params:
                try:
                    val = float(params['min_speed'])
                    query += " AND speed_kmh >= %s"
                    sql_params.append(val)
                except ValueError:
                    pass

            if 'max_speed' in params:
                try:
                    val = float(params['max_speed'])
                    query += " AND speed_kmh <= %s"
                    sql_params.append(val)
                except ValueError:
                    pass

            if 'direction' in params and params['direction']:
                query += " AND LOWER(COALESCE(direction, '')) = %s"
                sql_params.append(params['direction'].lower())

            if 'object_id' in params and params['object_id']:
                query += " AND COALESCE(object_id, '') LIKE %s"
                sql_params.append(f"%{params['object_id']}%")

            if 'start_date' in params and params['start_date']:
                query += " AND DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) >= %s"
                sql_params.append(params['start_date'])

            if 'end_date' in params and params['end_date']:
                query += " AND DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) <= %s"
                sql_params.append(params['end_date'])

            if 'min_confidence' in params:
                try:
                    val = float(params['min_confidence'])
                    query += " AND confidence >= %s"
                    sql_params.append(val)
                except ValueError:
                    pass

            if 'max_confidence' in params:
                try:
                    val = float(params['max_confidence'])
                    query += " AND confidence <= %s"
                    sql_params.append(val)
                except ValueError:
                    pass

            if 'reviewed_only' in params and params['reviewed_only'] == '1':
                query += " AND reviewed = 1"
            if 'flagged_only' in params and params['flagged_only'] == '1':
                query += " AND flagged = 1"
            if 'unannotated_only' in params and params['unannotated_only'] == '1':
                query += " AND reviewed = 0 AND flagged = 0"

            query += " ORDER BY datetime DESC LIMIT 1000"

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute(query, sql_params)
                rows = cursor.fetchall()

            data = [dict(row) for row in rows]
            speed_values = [float(d.get("speed_kmh") or 0) for d in data if d.get("speed_kmh") not in [None, ""]]
            avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0

            speeds = [float(d.get("speed_kmh") or 0) for d in data if d.get("speed_kmh") not in [None, ""]]
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
                "last_detection": data[0].get("datetime") if data else "N/A"
            }
            summary["speed_limits"] = load_config().get("dynamic_speed_limits", {})
            charts = {}
            try:
                resp = requests.get("http://127.0.0.1:5000/api/charts?days=0")
                if resp.ok:
                    charts = resp.json()
            except Exception as e:
                logger.warning(f"[Chart Fetch Error] {e}")

            config = load_config()
            summary["speed_limits"] = config.get("dynamic_speed_limits", {})
            logo_path = "/home/pi/radar/static/essi_logo.jpeg"
            filename = f"radar_filtered_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join("backups", filename)
            os.makedirs("backups", exist_ok=True)

            generate_pdf_report(filepath, data=data, summary=summary, filters=filters, logo_path=logo_path, charts=charts)
            return send_file(filepath, as_attachment=True)

        except Exception as e:
            logger.error(f"[EXPORT_FILTERED_PDF_ERROR] {e}")
            return str(e), 500
    
    @app.route("/retrain_model", methods=["POST"])
    @login_required
    def retrain_model():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            logger.info("[MODEL] Retraining LightGBM model from DB...")
            result = subprocess.run(["python3", "train_lightgbm.py"], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("[MODEL] Retraining completed successfully.")
                # Extract reported accuracy from stdout
                for line in result.stdout.splitlines():
                    if line.startswith("ACCURACY:"):
                        acc = float(line.strip().split(":")[1])
                        save_model_metadata(acc, "retrain")
                        break
                return jsonify({"status": "ok", "message": "Model retrained successfully."})
            else:
                logger.error(f"[MODEL] Retrain failed: {result.stderr}")
                return jsonify({"error": "Retraining failed.", "details": result.stderr}), 500
        except Exception as e:
            logger.exception(f"[RETRAIN ERROR] {e}")
            return jsonify({"error": "Internal server error."}), 500

    @app.route("/upload_model", methods=["POST"])
    @login_required
    def upload_model():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        if "model_file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["model_file"]

        if not file.filename.endswith(".pkl"):
            return jsonify({"error": "File must be a .pkl"}), 400

        try:
            from tempfile import NamedTemporaryFile
            import joblib

            with NamedTemporaryFile(delete=False) as tmp:
                file.save(tmp.name)
                loaded = joblib.load(tmp.name)

            if not isinstance(loaded, tuple) or len(loaded) != 2:
                return jsonify({"error": "Model format invalid. Expected (model, scaler) tuple."}), 400

            model, scaler = loaded

            if not isinstance(model, lgb.LGBMClassifier) or not isinstance(scaler, StandardScaler):
                return jsonify({"error": "Incorrect model or scaler type."}), 400

            joblib.dump((model, scaler), "radar_lightgbm_model.pkl")
            score = model.score(model.booster_.data, model.booster_.label)
            save_model_metadata(round(score * 100, 2), "upload")
            return jsonify({"status": "ok", "message": "Model uploaded and validated successfully."})

        except Exception as e:
            logger.exception("[MODEL UPLOAD ERROR]")
            return jsonify({"error": "Upload failed", "details": str(e)}), 500
    
    @app.route("/control", methods=["GET", "POST"])
    @login_required
    def control():
        if not is_admin():
            flash("Admin access required", "error")
            return redirect(url_for("index"))
        
        message = None
        config = load_config()
        try:
            cameras, selected = load_cameras_from_db()
            for cam in cameras:
                if "stream_type" not in cam:
                    cam["stream_type"] = "mjpeg"  # default fallback
            config["cameras"] = cameras
            config["selected_camera"] = selected
        except Exception as e:
            logger.warning(f"Could not load cameras from DB: {e}")
            config["cameras"] = []
            config["selected_camera"] = 0
        snapshot = None
        
        if request.method == "POST":
            action = request.form.get("action")
            
            try:
                if action == "clear_db":
                    with get_db_connection() as conn:
                        conn.execute("DELETE FROM radar_data")
                        conn.commit()
                    message = "All radar data cleared successfully."
                    
                elif action == "backup_db":
                    try:
                        backup_name = f"radar_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                        backup_path = os.path.join(BACKUP_FOLDER, backup_name)
                        os.makedirs(BACKUP_FOLDER, exist_ok=True)

                        result = subprocess.run(
                            ["pg_dump", "-U", "radar_user", "-h", "localhost", "-d", "radar_db", "-f", backup_path],
                            env={**os.environ, "PGPASSWORD": "securepass123"},
                            check=True
                        )

                        return send_file(backup_path, as_attachment=True, download_name=backup_name)

                    except Exception as e:
                        logger.error(f"[BACKUP ERROR] {e}")
                        message = f"Database backup failed: {str(e)}"
                    
                elif action == "restore_db":
                    if 'backup_file' in request.files:
                        file = request.files['backup_file']
                        if file and file.filename.endswith('.sql'):
                            filename = secure_filename(file.filename)
                            temp_path = os.path.join(BACKUP_FOLDER, f"temp_{filename}")
                            file.save(temp_path)

                            try:
                                subprocess.run(
                                    ["psql", "-U", "radar_user", "-h", "localhost", "-d", "radar_db", "-f", temp_path],
                                    env={**os.environ, "PGPASSWORD": "securepass123"},
                                    check=True
                                )
                                os.remove(temp_path)
                                message = "Database restored successfully."
                            except subprocess.CalledProcessError as e:
                                message = f"Restore failed: {e}"
                                logger.error(f"[RESTORE ERROR] {e}")
                        else:
                            message = "Please upload a valid .sql file."
                            
                elif action == "cleanup_snapshots":
                    retention_days = int(request.form.get("retention_days", config.get("retention_days", 30)))
                    cutoff = datetime.now() - timedelta(days=retention_days)
                    
                    with get_db_connection() as conn:
                        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                        cursor.execute("SELECT snapshot_path FROM radar_data WHERE COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp)) < %s", 
                                     (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
                        old_paths = [row['snapshot_path'] for row in cursor.fetchall() if row['snapshot_path']]
                        
                        deleted_count = 0
                        for path in old_paths:
                            if os.path.exists(path):
                                try:
                                    os.remove(path)
                                    deleted_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to delete {path}: {e}")
                        
                        cursor.execute("DELETE FROM radar_data WHERE COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp)) < %s", 
                                     (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
                        deleted_records = cursor.rowcount
                        conn.commit()
                    
                    message = f"Cleaned up {deleted_count} snapshots and {deleted_records} records."

                elif action == "test_radar":
                    try:
                        result = check_radar_connection(port="/dev/ttyACM0", baudrate=9600)
                        if result:
                            message = "Radar test successful. Connection established."
                            result.close()
                        else:
                            message = "Radar test failed. Device not responding on /dev/ttyACM0."
                    except Exception as e:
                        logger.error(f"[RADAR TEST] {e}")
                        message = f"Radar test error: {e}"
                    
                elif action == "test_camera":
                    try:
                        selected = config.get("selected_camera", 0)
                        cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
                        url = cam.get("url", "")
                        username = cam.get("username", "")
                        password = cam.get("password", "")
                        stream_type = cam.get("stream_type", "mjpeg")
                        auth = HTTPDigestAuth(username, password) if username and password else None

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        test_filename = f"test_{timestamp}.jpg"
                        test_path = os.path.join(SNAPSHOT_FOLDER, test_filename)
                        response = None

                        if stream_type == "snapshot":
                            response = requests.get(url, auth=auth, timeout=5)
                            if response.status_code == 200 and response.content.startswith(b'\xff\xd8'):
                                with open(test_path, "wb") as f:
                                    f.write(response.content)

                        elif stream_type == "mjpeg":
                            r = requests.get(url, auth=auth, stream=True, timeout=5)
                            buffer = b""
                            for chunk in r.iter_content(1024):
                                buffer += chunk
                                start = buffer.find(b'\xff\xd8')
                                end = buffer.find(b'\xff\xd9')
                                if start != -1 and end != -1 and end > start:
                                    frame = buffer[start:end+2]
                                    with open(test_path, "wb") as f:
                                        f.write(frame)
                                    break
                            r.close()

                        elif stream_type == "rtsp":
                            result = subprocess.run([
                                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                                "-i", url,
                                "-vframes", "1", "-q:v", "2", test_path
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)

                        if os.path.exists(test_path) and os.path.getsize(test_path) > 1024:
                            snapshot = os.path.basename(test_path)
                            message = "Camera test successful. Snapshot captured."
                        else:
                            message = "Camera test failed — no image returned."

                    except Exception as e:
                        logger.error(f"[CAMERA TEST] {e}")
                        message = f"Camera test error: {e}"
                    
                elif action == "update_config":
                    config["cooldown_seconds"] = float(request.form.get("cooldown_seconds", 0.5))
                    config["retention_days"] = int(request.form.get("retention_days", 30))
                    config["selected_camera"] = int(request.form.get("selected_camera", 0))
                    config["annotation_conf_threshold"] = float(request.form.get("annotation_conf_threshold", 0.5))
                    config["label_format"] = request.form.get("label_format", "{type} | {speed:.1f} km/h")

                    # Parse all camera fields
                    cameras = []
                    i = 0
                    while True:
                        cam_url = request.form.get(f"camera_url_{i}")
                        if not cam_url:
                            break
                        cam_username = request.form.get(f"camera_username_{i}", "")
                        cam_password = request.form.get(f"camera_password_{i}", "")
                        cam_type = request.form.get(f"camera_stream_type_{i}", "mjpeg")
                        cameras.append({
                            "url": cam_url.strip(),
                            "username": cam_username.strip(),
                            "password": cam_password.strip(),
                            "stream_type": cam_type.strip()
                        })
                        i += 1

                    if cameras:
                        config["cameras"] = cameras
                        save_cameras_to_db(cameras, config.get("selected_camera", 0))

                    # Dynamic speed limits
                    updated_limits = {}
                    for key in config.get("dynamic_speed_limits", {}).keys():
                        form_key = f"speed_limit_{key}"
                        val = request.form.get(form_key)
                        if val:
                            try:
                                updated_limits[key] = float(val)
                            except ValueError:
                                pass  # retain old

                    if updated_limits:
                        config["dynamic_speed_limits"] = updated_limits

                    if save_config(config):
                        message = "Configuration updated successfully."
                    else:
                        message = "Failed to save configuration."
                        
                elif action == "validate_snapshots":
                    invalid_count = validate_snapshots()
                    message = f"Snapshot validation complete. {invalid_count} invalid paths cleaned."
                        
            except Exception as e:
                logger.error(f"Control action error: {e}")
                message = f"Action failed: {str(e)}"
        
        # Get system stats
        try:
            disk_usage = psutil.disk_usage('/')
            disk_free = disk_usage.free / (1024**3)
        except Exception:
            disk_free = 0

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("SELECT COUNT(*) FROM radar_data")
                total_records = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM radar_data WHERE snapshot_path IS NOT NULL")
                snapshot_records = cursor.fetchone()[0]
        except Exception:
            total_records = 0
            snapshot_records = 0

        try:
            radar_ok = check_radar_connection(port="/dev/ttyACM0", baudrate=9600) is not None
        except Exception:
            radar_ok = False

        cams = config.get("cameras", [])
        selected = config.get("selected_camera", 0)
        cam = cams[selected] if cams and selected < len(cams) else {}
        stream_type = cam.get("stream_type", "mjpeg")
        camera_ok = False

        should_check_camera = action != "test_camera" if request.method == "POST" else True

        if should_check_camera:
            try:
                url = cam.get("url", "")
                username = cam.get("username", "")
                password = cam.get("password", "")
                if stream_type == "rtsp":
                    if url.startswith("rtsp://") and "@" not in url and username and password:
                        url = url.replace("rtsp://", f"rtsp://{username}:{password}@")

                    logger.info(f"[CONTROL CAMERA TEST] RTSP URL: {url}")
                    result = subprocess.run(
                        ["ffmpeg", "-rtsp_transport", "tcp", "-i", url, "-t", "1", "-f", "null", "-"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    camera_ok = (result.returncode == 0)

                elif stream_type == "mjpeg":
                    r = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True, timeout=5)
                    if r.status_code == 200:
                        buffer = b""
                        for chunk in r.iter_content(1024):
                            buffer += chunk
                            if b'\xff\xd8' in buffer and b'\xff\xd9' in buffer:
                                camera_ok = True
                                break
                    r.close()

                elif stream_type == "snapshot":
                    r = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=5)
                    if r.status_code == 200 and r.content.startswith(b'\xff\xd8'):
                        camera_ok = True

                logger.info(f"[CONTROL CAMERA TEST RESULT] camera_ok = {camera_ok}")

            except Exception as e:
                logger.warning(f"[CONTROL CAMERA TEST] Unexpected failure: {e}")

        try:
            return render_template("control.html",
                message=message,
                config=config,
                disk_free=round(disk_free, 2),
                total_records=total_records,
                snapshot_records=snapshot_records,
                snapshot=snapshot,
                radar_status=radar_ok,
                camera_status=camera_ok,
                model_info=get_model_metadata()
            )
        except Exception as e:
            import traceback
            logger.error(f"[CONTROL PAGE ERROR] {e}\n{traceback.format_exc()}")
            return f"<pre>{traceback.format_exc()}</pre>", 500

    
    @app.route("/users", methods=["GET", "POST"])
    @login_required
    def users():
        if request.method == "POST":
            action = request.form.get("action")
            
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                    
                    if action == "add_user":
                        if not is_admin():
                            flash("Admin access required", "error")
                            return redirect(url_for("users"))

                        username = request.form.get("username", "").strip()
                        password = request.form.get("password", "").strip()
                        role = request.form.get("role", "viewer")

                        if not username or not password:
                            flash("Username and password are required", "error")
                        elif len(password) < 6:
                            flash("Password must be at least 6 characters", "error")
                        elif role not in ["admin", "viewer"]:
                            flash("Invalid role", "error")
                        else:
                            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                                        (username, generate_password_hash(password), role))
                            conn.commit()
                            flash(f"User '{username}' added successfully.", "success")


                    elif action == "change_password":
                        current_password = request.form.get("current_password", "")
                        new_password = request.form.get("new_password", "")
                        confirm_password = request.form.get("confirm_password", "")

                        user = get_user_by_id(current_user.id)

                        if not all([current_password, new_password, confirm_password]):
                            flash("All password fields are required", "error")
                        elif new_password != confirm_password:
                            flash("New passwords do not match", "error")
                        elif len(new_password) < 6:
                            flash("New password must be at least 6 characters", "error")
                        elif not user or not check_password_hash(user.password_hash, current_password):
                            flash("Current password is incorrect", "error")
                        else:
                            cursor.execute(
                                "UPDATE users SET password_hash = %s WHERE id = %s",
                                (generate_password_hash(new_password), user.id)
                            )
                            conn.commit()
                            flash("Password changed successfully", "success")


            except psycopg2.IntegrityError:
                flash("Username already exists.", "error")
            except Exception as e:
                logger.error(f"Error managing users: {e}")
                flash("Error managing users.", "error")

            return redirect(url_for("users"))

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                # Get all users and their last activity
                cursor.execute("""
                    SELECT u.id, u.username, u.role, u.created_at,
                        ua.last_activity
                    FROM users u
                    LEFT JOIN user_activity ua ON u.id = ua.user_id
                    ORDER BY u.username
                """)
                users_data = cursor.fetchall()

                active_user_ids = set()
                cutoff_time = datetime.now() - timedelta(minutes=30) 

                for user in users_data:
                    ts = user['last_activity']
                    if ts:
                        try:
                            # Ensure ts is always timezone-aware
                            if isinstance(ts, str):
                                ts = datetime.fromisoformat(ts)
                            elif not isinstance(ts, datetime):
                                continue

                            if ts >= cutoff_time:
                                active_user_ids.add(user['id'])
                        except Exception as e:
                            logger.warning(f"Failed to parse timestamp for user {user['username']}: {e}")


                # Mark each user
                users_list = []
                for user in users_data:
                    user_dict = dict(user)
                    user_dict['is_active'] = user['id'] in active_user_ids
                    users_list.append(user_dict)

                logger.info(f"Loaded {len(users_list)} users, {len(active_user_ids)} active.")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            flash("Error loading users", "error")
            return render_template("users.html", users=[])
    
        return render_template("users.html", users=users_list)
    
    @app.route('/delete_user/<int:user_id>', methods=['POST'])
    @login_required
    def delete_user(user_id):
        if not current_user.is_authenticated:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({'success': False, 'error': 'Session expired'}), 401
            else:
                return redirect(url_for('login'))

        if current_user.role != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        if current_user.id == user_id:
            return jsonify({'success': False, 'error': 'You cannot delete your own account'}), 400

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()

                if not user:
                    return jsonify({'success': False, 'error': 'User not found'}), 404

                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
                cursor.execute("DELETE FROM user_activity WHERE user_id = %s", (user_id,))
                conn.commit()

            return jsonify({'success': True}), 200

        except Exception as e:
            logger.exception(f"[DELETE USER ERROR] {e}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route("/change_password", methods=["POST"])
    @login_required
    def change_password():
        try:
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not all([current_password, new_password, confirm_password]):
                flash("All fields are required", "error")
                return redirect(url_for("users"))

            if new_password != confirm_password:
                flash("New passwords do not match", "error")
                return redirect(url_for("users"))

            if len(new_password) < 6:
                flash("New password must be at least 6 characters", "error")
                return redirect(url_for("users"))

            user = get_user_by_id(current_user.id)
            if not user or not check_password_hash(user.password_hash, current_password):
                flash("Current password is incorrect", "error")
                return redirect(url_for("users"))

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                               (generate_password_hash(new_password), user.id))
                conn.commit()

            flash("Password changed successfully", "success")
            return redirect(url_for("users"))
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            flash("Error changing password", "error")
            return redirect(url_for("users"))
    
    @app.route("/api/active_users")
    @login_required
    def api_active_users():
        try:
            active_users = get_active_users(30)
            return jsonify({
                "active_count": len(active_users),
                "active_users": [
                    {
                        "username": user['username'],
                        "role": user['role'],
                        "last_activity": user['last_activity']
                    } for user in active_users
                ]
            })
        except Exception as e:
            logger.error(f"Error getting active users API: {e}")
            return jsonify({"error": "Internal server error"}), 500
        
    @app.before_request
    def before_request():
        if current_user.is_authenticated:
            update_user_activity(current_user.id)
    
    return app

# Ensure app is bootstrapped
ensure_directories()
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


