import serial
import time

class OPS243CRadar:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False

    def connect(self, retries=5, delay=1.0):
        import subprocess
        for attempt in range(retries):
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                    time.sleep(0.2)

                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                if self.ser.is_open:
                    print(f"Radar connected on {self.port}")

                    # Reset buffers
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    time.sleep(0.1)          

                    return True

            except Exception as e:
                print(f"[Radar Connect] Attempt {attempt + 1} failed: {e}")
                time.sleep(delay)
                delay *= 1.5

                if attempt >= 2 and self.port.startswith("/dev/ttyACM"):
                    print("[Radar Connect] Attempting USB reset...")
                    subprocess.call(["usbreset", self.port])  

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
