import time
import json
from scipy.signal import find_peaks
from acquire_radar_data import OPS243CRadar
from classify_objects import ObjectClassifier
from kalman_filter_tracking import ObjectTracker
from collections import deque
import numpy as np

DOPPLER_SCALE = 3e8 / (2 * 24e9)  # ~6.25 for 24 GHz radar
FRAME_INTERVAL = 0.2  # seconds

DISTANCE_RES = 0.05  # meters per FFT bin
SIGNAL_THRESHOLD = 15
MAX_OBJECTS = 5

class MultiObjectProcessor:
    def __init__(self):
        self.tracker = ObjectTracker()
        self.classifier = ObjectClassifier()
        self.fft_history = deque(maxlen=2)
        self.prev_peaks = []
        self.prev_time = None

    def parse_fft(self, line):
        try:
            if line.startswith('{') and '"FFT"' in line:
                data = json.loads(line)
                return data['FFT']
        except json.JSONDecodeError:
            return None
        return None
    
    def estimate_velocities_from_peaks(self, current_peaks, current_time):
        velocities = {}

        if not self.prev_peaks or self.prev_time is None:
            self.prev_peaks = current_peaks
            self.prev_time = current_time
            return velocities

        dt = current_time - self.prev_time
        if dt <= 0:
            return velocities

        for cp in current_peaks:
            closest_prev = min(self.prev_peaks, key=lambda pp: abs(pp - cp), default=None)
            if closest_prev is not None and abs(cp - closest_prev) <= 2:
                delta_bins = cp - closest_prev
                doppler_shift = delta_bins / dt
                velocity = doppler_shift * DOPPLER_SCALE
                velocities[cp] = velocity

        self.prev_peaks = current_peaks
        self.prev_time = current_time
        return velocities

    def extract_objects_from_fft(self, fft_data, current_time):
        peaks, _ = find_peaks(fft_data, height=SIGNAL_THRESHOLD, distance=2)
        velocities = self.estimate_velocities_from_peaks(peaks, current_time)

        objects = []
        for p in peaks[:MAX_OBJECTS]:
            signal = fft_data[p]
            distance = p * DISTANCE_RES
            velocity = velocities[p] if p in velocities else None

            obj = {
                'radar_distance': distance,
                'distance': distance,
                'speed_kmh': abs(velocity) * 3.6 if velocity is not None else 0.0,
                'velocity': velocity if velocity is not None else 0.0,
                'doppler_frequency': velocity / DOPPLER_SCALE if velocity else 0.0,
                'signal_level': signal,
                'sensor': 'OPS243C',
                'confidence': 1.0
            }
            objects.append(obj)
        return objects

    def run(self):
        radar = OPS243CRadar(port='COM7', baudrate=9600)
        if not radar.connect():
            print("[ERROR] Could not connect to radar.")
            return

        radar.configure_radar()
        radar.running = True

        print("Starting multi-object radar tracking...")

        try:
            while True:
                raw_data = radar.read_data(timeout=0.2)
                all_fft = []
                for line in raw_data:
                    fft = self.parse_fft(line)
                    if fft:
                        all_fft.extend(fft)

                if all_fft:
                    current_time = time.time()
                    detected_objects = self.extract_objects_from_fft(all_fft, current_time)
                    classified = self.classifier.classify_objects(detected_objects)
                    tracked = self.tracker.update_tracks(classified, frame_timestamp=current_time)

                    for obj in tracked:
                        print("\n--- Current Detection ---")
                        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(obj['timestamp']))}")
                        print(f"Object ID: {obj['object_id']}")
                        print(f"Sensor: {obj['sensor']}")
                        print(f"Type: {obj['type']}")
                        print(f"Confidence: {obj['confidence']:.2f}")
                        print(f"Distance: {obj['distance']:.2f} m")
                        print(f"Velocity: {obj['velocity']:.2f} m/s")
                        print(f"Speed: {obj['speed_kmh']:.2f} km/h")
                        print(f"Signal Level: {obj['signal_level']:.1f}")
                        print(f"Doppler Frequency: {obj['doppler_frequency']:.2f} Hz")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nStopping radar...")
            radar.stop()
            print("Radar stopped.")

if __name__ == "__main__":
    processor = MultiObjectProcessor()
    processor.run()
