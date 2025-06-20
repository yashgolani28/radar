import time
import json
import os
import cv2
from acquire_radar_data import OPS243CRadar
from process_radar_data import ProcessRadarData
from kalman_filter_tracking import ObjectTracker
from classify_objects import ObjectClassifier
from camera import capture_snapshot
from bounding_box import annotate_speeding_object

def main():
    print("[DEBUG] Starting radar detection system with enhanced output and classification...")

    # Initialize radar
    radar = OPS243CRadar(port='COM7', baudrate=9600)
    if not radar.connect():
        print("[ERROR] Failed to connect to radar")
        return

    radar.configure_radar()
    radar.running = True

    # Initialize object tracking and classification
    tracker = ObjectTracker(speed_limit_kmh=1)
    classifier = ObjectClassifier()

    # Camera configuration
    CAMERA_URL = "http://192.168.1.109/axis-cgi/jpg/image.cgi"
    CAMERA_USERNAME = "root"
    CAMERA_PASSWORD = "2024"
    snapshot_dir = "snapshots/"  # Ensure snapshots are saved here

    # Ensure the snapshot directory exists
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
        print(f"[DEBUG] Created directory: {snapshot_dir}")

    try:
        while True:
            raw_data = radar.read_data(timeout=0.2)
            object_count = None

            if raw_data:
                for line in raw_data:
                    try:
                        data = json.loads(line)
                        if 'DetectedObjectCount' in data:
                            object_count = data['DetectedObjectCount']
                    except json.JSONDecodeError:
                        pass

                # Process the radar data
                processor = ProcessRadarData(raw_data)
                detection = processor.analyze()

                if detection:
                    current_time = time.time()
                    data_point = {
                        'timestamp': current_time,
                        'sensor': 'OPS243C',
                        'distance': detection['distance'],
                        'velocity': detection['velocity'],
                        'speed_kmh': detection['speed_kmh'],
                        'confidence': 1.0
                    }

                    # Classify detected objects and update tracking
                    classified = classifier.classify_objects([data_point])
                    tracked = tracker.update_tracks(classified)

                    if object_count is not None:
                        print(f"\n[INFO] Detected Object Count: {object_count}")

                    for obj in tracked:
                        # Print basic detection details
                        print("\n--- Current Detection ---")
                        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(obj['timestamp']))}")
                        print(f"Object ID: {obj['object_id']}")
                        print(f"Sensor: {obj['sensor']}")
                        print(f"Type: {obj['type']}")
                        print(f"Confidence: {obj['confidence']:.2f}")
                        print(f"Speed Limit: {tracker.speed_limit_kmh:.2f} km/h")
                        print(f"Distance: {detection['distance']:.2f} m")
                        print(f"Velocity: {detection['velocity']:.2f} m/s")
                        print(f"Speed: {detection['speed_kmh']:.2f} km/h")
                        print(f"Signal Level: {detection['signal_level']:.1f} dB")
                        print(f"Doppler Frequency: {detection['doppler_frequency']:.2f} Hz")

                        if detection.get('direction') == "approaching":
                            print("Status: 🔴 APPROACHING")
                        elif detection.get('direction') == "departing":
                            print("Status: 🟡 DEPARTING")
                        else:
                            print("Status: ⚪ STATIONARY")

                        # Check if speeding violation occurs
                        if obj['speed_kmh'] > tracker.speed_limit_kmh:
                            print(f"[ALERT] Speeding detected: {obj['speed_kmh']:.2f} km/h")

                            # Capture snapshot if speeding detected
                            snapshot_path = capture_snapshot(
                                camera_url=CAMERA_URL,
                                username=CAMERA_USERNAME,
                                password=CAMERA_PASSWORD
                            )

                            # Check if snapshot path is returned
                            if snapshot_path:
                                print(f"[CAMERA] Snapshot captured: {snapshot_path}")
                                try:
                                    # Annotate the snapshot with bounding box and speed
                                    annotated_path = annotate_speeding_object(
                                        image_path=snapshot_path,
                                        radar_distance=obj['distance'],
                                        label=f"{obj['speed_kmh']:.1f} km/h",
                                        save_dir=snapshot_dir
                                    )
                                    if annotated_path:
                                        print(f"[ANNOTATION] Annotated image saved: {annotated_path}")
                                    else:
                                        print("[WARNING] No bounding box created.")
                                except Exception as e:
                                    print(f"[ERROR] Failed to annotate snapshot: {e}")
                            else:
                                print("[ERROR] Snapshot capture failed.")

                else:
                    print("[WARNING] No valid detection")

            else:
                print("[INFO] Waiting for more data...")

            # Sleep before next radar reading
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping radar...")
        radar.stop()
        print("[INFO] Radar stopped.")

if __name__ == "__main__":
    main()
