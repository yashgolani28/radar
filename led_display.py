from rgbmatrix import RGBMatrix, RGBMatrixOptions, graphics
import time

class SpeedDisplay:
    def __init__(self):
        options = RGBMatrixOptions()
        options.rows = 64
        options.cols = 64
        options.chain_length = 1
        options.parallel = 1
        options.hardware_mapping = 'adafruit-hat-pwm'
        options.panel_type = 'FM6126A'
        options.scan_mode = 1
        options.brightness = 100
        self.matrix = RGBMatrix(options=options)
        self.canvas = self.matrix.CreateFrameCanvas()
        self.font = graphics.Font()
        self.font.LoadFont("/home/pi/matrix-full/fonts/6x10.bdf")
        self.color_map = {
            "CAR": graphics.Color(255, 0, 0),
            "HUMAN": graphics.Color(0, 255, 0),
            "BIKE": graphics.Color(255, 255, 0),
            "TRUCK": graphics.Color(255, 128, 0),
            "BICYCLE": graphics.Color(128, 255, 255),
            "UNKNOWN": graphics.Color(128, 128, 128)
        }
        self.default_color = graphics.Color(255, 255, 255)
        self.last_message = ""
        self.last_update_time = time.time()
        self.idle_timeout = 10  # seconds

    def display_speed(self, obj_type: str, speed: float):
        self.canvas.Clear()
        graphics.DrawText(self.canvas, self.font, 2, 20, graphics.Color(0, 255, 0), "LINE 1")
        graphics.DrawText(self.canvas, self.font, 2, 34, graphics.Color(255, 0, 0), "LINE 2")
        graphics.DrawText(self.canvas, self.font, 2, 48, graphics.Color(0, 0, 255), "LINE 3")
        self.canvas = self.matrix.SwapOnVSync(self.canvas)

        text = f"{obj_type}: {speed:.1f} km/h"
        color = self.color_map.get(obj_type.upper(), self.default_color)
        self.scroll_text(text, color)
        self.last_message = text
        self.last_update_time = time.time()

    def show_idle(self, message="Waiting for detection..."):
        if self.last_message != message:
            self.canvas.Clear()
            graphics.DrawText(self.canvas, self.font, 2, 34, graphics.Color(100, 100, 255), message)
            self.canvas = self.matrix.SwapOnVSync(self.canvas)
            self.last_message = message
            self.last_update_time = time.time()

    def scroll_text(self, text, color):
        self.canvas.Clear()
        pos = self.canvas.width
        while pos + len(text) * 6 > 0:
            self.canvas.Clear()
            graphics.DrawText(self.canvas, self.font, pos, 34, color, text)
            pos -= 1
            self.canvas = self.matrix.SwapOnVSync(self.canvas)
            time.sleep(0.03)

    def auto_clear_if_idle(self):
        if time.time() - self.last_update_time > self.idle_timeout:
            self.canvas.Clear()
            self.canvas = self.matrix.SwapOnVSync(self.canvas)
            self.last_message = ""
            self.last_update_time = time.time()
    
    def show_text(self, message: str):
        self._display_text(message.upper())

