import re
import time

class ProcessRadarData:
    def __init__(self, raw_data, distance_scale=1.0):
        self.raw_data = raw_data
        self.distance_scale = distance_scale
        self.radar_freq = 24e9
        self.speed_of_light = 3e8

    def analyze(self):
        if not self.raw_data:
            return None
        
        detection = {
            "timestamp": time.time(),
            "frame_count": 0,
            "distance": 0.0,
            "velocity": 0.0,
            "speed_kmh": 0.0,
            "direction": "stationary",
            "doppler_frequency": 0.0,
            "signal_level": 0.0
        }

        for line in reversed(self.raw_data):
            temp_detection = self._parse_line(line)
            if temp_detection:
                detection.update(temp_detection)

                if 'distance' in detection and 'velocity' in detection:
                    detection['signal_level'] = self._calc_signal_level(detection['distance'])
                    detection['doppler_frequency'] = self._calc_doppler_freq(detection['distance'])
                    detection['speed_kmh'] = abs(detection['velocity']) * 3.6

                    if detection['velocity'] > 0:
                        detection['direction'] = "approaching"
                    elif detection['velocity'] < 0:
                        detection['direction'] = "departing"
                    else:
                        detection['direction'] = "stationary"

        return detection

    def _parse_line(self, line):
        parsed_data = {}

        # Match distance like "m",2.0
        match_distance = re.search(r'"m",\s*(-?\d*\.?\d+)', line)
        if match_distance:
            parsed_data["distance"] = float(match_distance.group(1)) * self.distance_scale

        # Match velocity like "mps",-0.4
        match_velocity = re.search(r'"mps",\s*(-?\d*\.?\d+)', line)
        if match_velocity:
            parsed_data["velocity"] = float(match_velocity.group(1))

       # print(f"[DEBUG] Parsed line: {line.strip()} → {parsed_data}")
        return parsed_data

    def _calc_signal_level(self, distance):
        return max(10, 100 - (distance * 2))

    def _calc_doppler_freq(self, distance):
        return (2 * self.radar_freq * distance) / self.speed_of_light
