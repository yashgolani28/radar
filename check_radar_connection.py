import serial

def check_radar_connection(port="COM7", baudrate=9600):
    try:
        # Attempt to open the serial connection
        radar_connection = serial.Serial(port, baudrate, timeout=1)
        radar_connection.flushInput()
        print(f"Radar connected on {port} at {baudrate} baud rate.")
        return radar_connection
    except Exception as e:
        print(f"Failed to connect to radar: {e}")
        return None
