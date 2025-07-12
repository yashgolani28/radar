import time
import json
import os
import logging
from datetime import datetime
import psycopg2
import psycopg2.extras
from collections import deque
import cv2
import numpy as np
from acquire_radar_data import OPS243CRadar
from process_radar_data import ProcessRadarData
from multi_object_tracking import MultiObjectProcessor
from kalman_filter_tracking import ObjectTracker
from classify_objects import ObjectClassifier
from camera import capture_snapshot
from bounding_box import annotate_speeding_object, annotate_async
from logger import logger  
from threading import Thread
from collections import defaultdict, deque
from config_utils import load_config # type: ignore

config = load_config()
tracker = ObjectTracker(
    speed_limit_kmh=config.get("dynamic_speed_limits", {}).get("default", 3.0),
    speed_limits_map=config.get("dynamic_speed_limits", {})
)
frame_buffer = deque(maxlen=6)
speeding_buffer = defaultdict(lambda: deque(maxlen=5))
_last_reload_time = 0


def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def should_reload_config():
    global _last_reload_time
    try:
        flag_path = "reload_flag.txt"
        if not os.path.exists(flag_path):
            return False

        modified = os.path.getmtime(flag_path)

        # Only trigger if the flag file is newer than last reload
        if modified > _last_reload_time:
            _last_reload_time = time.time()  # update local memory
            logger.info("[CONFIG] Detected config reload request.")
            return True

        return False
    except Exception as e:
        logger.warning(f"[CONFIG RELOAD CHECK ERROR] {e}")
        return False

def main():
    global config  # use the global config loaded at top
    logger.info("Starting radar detection system with enhanced output and classification...")

    radar = OPS243CRadar(port='/dev/ttyACM0', baudrate=9600)
    if not radar.connect():
        logger.critical("Failed to connect to radar")
        return

    radar.configure_radar()
    radar.running = True
    
    classifier = ObjectClassifier()
    last_snapshot_ids = {}
    last_snapshot_times = {}  
    COOLDOWN_SECONDS = 0.05

    CAMERA_URL = "http://192.168.1.241/axis-cgi/jpg/image.cgi?resolution=640x480"
    CAMERA_USERNAME = "root"
    CAMERA_PASSWORD = "2024"

    error_count = 0
    max_errors = 5

    try:
        while True:
            if should_reload_config():
                logger.info("[CONFIG] Reloading config from disk")
                config = load_config()
                tracker.speed_limits_map = config.get("dynamic_speed_limits", {})
            raw_data = None
            try:
                raw_data = radar.read_data(timeout=0.2)
            except Exception as e:
                logger.warning(f"Radar read error: {e}")
                error_count += 1
                if error_count >= max_errors:
                    logger.error("Too many read failures, attempting radar reset...")
                    radar.stop()
                    time.sleep(1)
                    if not radar.connect():
                        logger.critical("Radar reconnect failed. Exiting main loop.")
                        break
                    radar.configure_radar()
                    error_count = 0
                continue

            if not raw_data:
                logger.warning("No data received from radar")
                continue

            try:
                object_count = None
                for line in raw_data:
                    try:
                        data = json.loads(line)
                        if 'DetectedObjectCount' in data:
                            object_count = data['DetectedObjectCount']
                    except json.JSONDecodeError:
                        pass

                processor_fft = MultiObjectProcessor()
                processor_native = ProcessRadarData(raw_data)
                native_detection = processor_native.analyze()

                fft_data = []
                for line in raw_data:
                    fft = processor_fft.parse_fft(line)
                    if fft:
                        fft_data.extend(fft)

                fft_detections = []
                if fft_data:
                    current_time = time.time()
                    fft_detections = processor_fft.extract_objects_from_fft(fft_data, current_time)

                fused_objects = []

                if native_detection:
                    native_detection["source"] = "native"
                    native_detection["sensor"] = "OPS243C"
                    native_detection["confidence"] = 1.0
                    fused_objects.append(native_detection)

                for fft_obj in fft_detections:
                    matched = False
                    for native in fused_objects:
                        d_native = native.get("radar_distance", native.get("distance", 0.0))
                        d_fft = fft_obj.get("radar_distance", 0.0)
                        if abs(d_native - d_fft) < 0.3:
                            fft_v = fft_obj.get("velocity", 0.0)
                            native_v = native.get("velocity", 0.0)

                            # Only replace if native velocity is weak and FFT velocity is confident
                            if abs(native_v) < 0.2 and abs(fft_v) > 0.3:
                                logger.debug(f"[FUSION] Overriding native velocity {native_v:.2f} with FFT {fft_v:.2f}")
                                native["velocity"] = fft_v
                                native["source"] = "fused"
                                native["fusion_flag"] = "fft_overwrite"

                            matched = True
                            break
                    if not matched:
                        fft_obj["source"] = "fft"
                        fft_obj["distance"] = fft_obj["radar_distance"]
                        fused_objects.append(fft_obj)

                if fused_objects:
                    current_time = time.time()
                    classified = classifier.classify_objects(fused_objects)
                    yolo_detections = []

                    for obj in classified:
                        obj["timestamp"] = current_time
                        obj["sensor"] = obj.get("sensor", "OPS243C")
                        obj["signal_level"] = obj.get("signal_level", 0.0)
                        obj["doppler_frequency"] = obj.get("doppler_frequency", 0.0)
                        obj["visual_distance"] = 0.0
                        obj["distance"] = obj.get("radar_distance", obj.get("distance", 0.0))
                        obj["direction"] = obj.get("direction", "unknown")
                        obj["raw_velocity"] = obj.get("velocity", 0.0)

                        obj["radar_distance"] = obj.get("radar_distance", obj.get("distance", 0.0))
                        obj["distance"] = obj["radar_distance"]

                    tracked = tracker.update_tracks(
                        detections=classified,
                        yolo_detections=yolo_detections,
                        frame_timestamp=current_time
                    )

                    if object_count is not None:
                        logger.info(f"[INFO] Detected Object Count: {object_count}")

                    for obj in tracked:
                        logger.info("\n--- Current Detection ---")
                        logger.info(f"Timestamp: {datetime.fromtimestamp(obj['timestamp'])}")
                        logger.info(f"Object ID: {obj['object_id']}")
                        logger.info(f"Sensor: {obj['sensor']}")
                        logger.info(f"Type: {obj['type']}")
                        logger.info(f"Confidence: {obj['confidence']:.2f}")
                        logger.info(f"Speed Limit: {tracker.speed_limit_kmh:.2f} km/h")
                        logger.info(f"Radar Distance: {obj['radar_distance']:.2f} m")
                        logger.info(f"Visual Distance: {obj['visual_distance']:.2f} m")
                        logger.info(f"Velocity: {obj['velocity']:.2f} m/s")
                        logger.info(f"Speed: {obj['speed_kmh']:.2f} km/h")
                        logger.info(f"Signal Level: {obj['signal_level']:.1f} dB")
                        logger.info(f"Doppler Frequency: {obj['doppler_frequency']:.2f} Hz")

                        if obj.get('direction') == "approaching":
                            logger.info("Status: 🔴 APPROACHING")
                        elif obj.get('direction') == "departing":
                            logger.info("Status: 🟡 DEPARTING")
                        else:
                            logger.info("Status: ⚪ STATIONARY")

                        score = obj.get("score", 0)
                        logger.info(f"[SCORE] {obj['object_id']} = {score:.2f}")

                        obj_id = obj['object_id']
                        now = time.time()
                        is_speeding = float(score) >= 0.8 and float(obj['speed_kmh']) > float(tracker.speed_limit_kmh)
                        speeding_buffer.setdefault(obj_id, deque(maxlen=5))

                        if is_speeding:
                            speeding_buffer[obj_id].append(now)
                        else:
                            speeding_buffer[obj_id].clear()

                        recent_events = speeding_buffer[obj_id]
                        should_trigger = False

                        speed_limit = tracker.get_limit_for(obj.get("type", "UNKNOWN"))
                        should_trigger = False

                        if obj['speed_kmh'] > speed_limit:
                            logger.info(f"[TRIGGER] Speeding: {obj['speed_kmh']:.1f} km/h > {speed_limit} km/h")
                            should_trigger = True
                        elif len(recent_events) >= 2 and (now - recent_events[0]) <= 2.0:
                            logger.info(f"[BUFFER] Repeated high-speed detections within window")
                            should_trigger = True

                        if should_trigger:
                            last_taken = last_snapshot_ids.get(obj_id, 0)
                            cooldown = config.get("cooldown_seconds", COOLDOWN_SECONDS)

                            if now - last_taken < cooldown:
                                logger.info(f"[SKIP] Snapshot cooldown active for {obj_id} ({now - last_taken:.2f}s < {cooldown}s)")
                            else:
                                last_snapshot_ids[obj_id] = now

                                # Capture fresh snapshot
                                raw_jpg_path = capture_snapshot(
                                    camera_url=CAMERA_URL,
                                    username=CAMERA_USERNAME,
                                    password=CAMERA_PASSWORD
                                )

                                if raw_jpg_path and os.path.exists(raw_jpg_path):
                                    try:
                                        frame = cv2.imread(raw_jpg_path)
                                        if frame is not None:
                                            sharpness = compute_sharpness(frame)
                                            frame_buffer.append({"image": frame, "path": raw_jpg_path, "sharpness": sharpness})
                                    except Exception as e:
                                        logger.warning(f"[FRAME CACHE ERROR] {e}")

                                if frame_buffer:
                                    best = max(frame_buffer, key=lambda f: f["sharpness"])
                                    snapshot_frame = best["image"]
                                    snapshot_path = best["path"]

                                    label_fmt = config.get("label_format", "{type} | {speed:.1f} km/h")
                                    label = label_fmt.format(type=obj['type'], speed=obj['speed_kmh'])

                                    logger.info("[ANNOTATION] Best snapshot selected, annotating...")
                                    try:
                                        annotated_path, visual_distance, updated_radar = annotate_speeding_object(
                                            image_path=snapshot_path,
                                            radar_distance=obj['radar_distance'],
                                            label=label,
                                            min_confidence=config.get("annotation_conf_threshold", 0.5)
                                        )

                                        if annotated_path:
                                            obj["visual_distance"] = visual_distance
                                            obj["radar_distance"] = updated_radar

                                        conn = psycopg2.connect(
                                            dbname="radar_db",
                                            user="radar_user",
                                            password="securepass123",
                                            host="localhost"
                                        )
                                        cursor = conn.cursor()

                                        cursor.execute("""
                                            INSERT INTO radar_data (
                                                timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                                                velocity, distance, radar_distance, visual_distance,
                                                direction, signal_level, doppler_frequency, snapshot_path
                                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                        """, (
                                            obj['timestamp'],
                                            datetime.fromtimestamp(obj['timestamp']).strftime("%Y-%m-%d %H:%M:%S"),
                                            obj['sensor'],
                                            obj['object_id'],
                                            obj['type'],
                                            obj['confidence'],
                                            obj['speed_kmh'],
                                            obj['velocity'],
                                            obj['distance'],
                                            obj['radar_distance'],
                                            obj['visual_distance'],
                                            obj['direction'],
                                            obj['signal_level'],
                                            obj['doppler_frequency'],
                                            snapshot_path
                                        ))
                                        conn.commit()
                                        conn.close()
                                    except Exception as e:
                                        logger.error(f"[ANNOTATE or DB ERROR] {e}")
                                    except Exception as e:
                                        logger.error(f"[ANNOTATE ERROR] {e}")
                                else:
                                    logger.error("[CAMERA] Snapshot failed.")
                        else:
                            logger.info("[FILTERED] Conditions not met for snapshot.")

                else:
                    logger.info("No valid object detected.")
            except Exception as loop_error:
                logger.exception(f"[Processing Error] {loop_error}")

            if not fused_objects:
                time.sleep(0.2)  
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping radar.")
        radar.stop()

if __name__ == "__main__":
    main()
