import time
import uuid
from collections import deque

class KalmanFilter:
    def __init__(self, initial_position):
        self.x = initial_position
        self.v = 0.0
        self.P = 1.0
        self.last_update = time.time()

    def update(self, measurement):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        if dt <= 0:
            dt = 1e-3

        A = 1
        Q = 0.01
        R = 0.1
        H = 1

        self.x += self.v * dt
        self.P += Q

        y = measurement - H * self.x
        S = H * self.P * H + R
        K = self.P * H / S

        self.x += K * y
        self.P = (1 - K * H) * self.P
        self.v = y / dt

    def get_state(self):
        return self.x, self.v


class ObjectTracker:
    def __init__(self, speed_limit_kmh=1):
        self.trackers = {}
        self.history = {}
        self.speed_limit_kmh = speed_limit_kmh
        self.max_age = 5  # seconds
        self.match_tolerance = 0.2  # meters

    def _generate_id(self, distance):
        bin_dist = round(distance / self.match_tolerance) * self.match_tolerance
        return f"obj_{bin_dist:.3f}_{uuid.uuid4().hex[:4]}"

    def update_tracks(self, detections):
        current_time = time.time()
        updated_objects = []

        # Clean old trackers
        self.trackers = {
            oid: tracker for oid, tracker in self.trackers.items()
            if current_time - tracker.last_update < self.max_age
        }

        # Match and update trackers
        for det in detections:
            matched_id = None
            for oid, tracker in self.trackers.items():
                pred_x, _ = tracker.get_state()
                if abs(pred_x - det['distance']) < self.match_tolerance:
                    matched_id = oid
                    break

            if not matched_id:
                matched_id = self._generate_id(det['distance'])
                self.trackers[matched_id] = KalmanFilter(det['distance'])
                self.history[matched_id] = deque(maxlen=5)

            tracker = self.trackers[matched_id]
            tracker.update(det['distance'])
            est_pos, est_vel = tracker.get_state()
            speed_kmh = abs(est_vel) * 3.6

            det.update({
                'object_id': matched_id,
                'distance': est_pos,
                'velocity': est_vel,
                'speed_kmh': speed_kmh,
                'timestamp': current_time
            })

            self.history[matched_id].append((current_time, est_pos))

            updated_objects.append(det)

        return updated_objects
