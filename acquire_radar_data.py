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
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                print("Radar connected.")
                return True
            else:
                print("Serial port opened but not readable.")
        except Exception as e:
            print(f"Failed to connect to radar: {e}")
        self.ser = None
        return False

    def configure_radar(self):
        if self.ser:
            commands = [
                b'F2\n',
               # b'r>0.5\n',
               # b'r<120\n',
               # b'R>1.0\n',
               # b'R<50\n',
                b'PX\n',
                b'R|\n',
                b'm?\n',
                b'R?\n',
                b'N?\n',
                b'OF\n'
            ]
            for cmd in commands:
                try:
                    self.ser.write(cmd)
                    time.sleep(0.05)  # Let radar process
                except Exception as e:
                    print(f"[RADAR CONFIG ERROR] Cmd {cmd} failed: {e}")

    def read_data(self, timeout=2):
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port not connected.")
        end_time = time.time() + timeout
        raw_data = []
        while time.time() < end_time:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                raw_data.append(line)
        return raw_data if raw_data else None

    def stop(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None
            print("Radar stopped.")
