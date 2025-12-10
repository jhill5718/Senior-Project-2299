# This program has continuous autofocus
# Code opens preview for 10s, then snaps a pic

from picamera2 import Picamera2
from libcamera import controls
import time

picam2 = Picamera2()
# print(picam2.camera_controls) # displays the dictionary values (min, max, default)

picam2.start(show_preview=True)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

time.sleep(10)
picam2.capture_file("test.jpg")
