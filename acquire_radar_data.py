import serial
import time

class OPS243CRadar:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            if self.ser.is_open:
                print("Radar connected.")
                return True
            else:
                print("Serial port opened but not readable.")
        except Exception as e:
            print(f"Failed to connect to radar: {e}")
        self.ser = None  # Ensure it's None if connection failed
        return False

    def configure_radar(self):
        if self.ser:
            commands = [
                b'F2\n',          # Precision
                #b'r>0.5\n',       # Ignore objects closer than 0.5 m
                #b'r<120\n',       # Ignore objects farther than 120 m
                #b'R>1.0\n',       # Ignore speeds below 1.0 m/s
                #b'R<50\n',        # Ignore speeds above 50 m/s
                #b'PX\n',          # Max transmit power
                b'R|\n',          # Report both directions
                b'm?\n',          # Query magnitude filter (optional)
                b'R?\n'           # Query current speed filter (optional)
                b'N?\n',          # Object sount
                b'OF\n'           # FFT
            ]

            for cmd in commands:
                self.ser.write(cmd)
                time.sleep(0.1)

            print("Radar configured with SRF mode, filters, and max power.")

    def read_data(self, timeout=2):
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port not connected.")
        end_time = time.time() + timeout
        raw_data = []
        while time.time() < end_time:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                raw_data.append(line)
        return raw_data if raw_data else None

    def stop(self):
        if self.ser:
            self.ser.close()
            print("Radar stopped.")
