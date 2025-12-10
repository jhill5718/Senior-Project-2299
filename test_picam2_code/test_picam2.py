# Keeps preview open for 10 seconds, then snaps a photo
from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.start_preview(Preview.QT)

config = picam2.create_preview_configuration()
picam2.configure(config)

picam2.start()
time.sleep(10)

picam2.capture_file("photo.jpg")

picam2.stop_preview()
picam2.close()
