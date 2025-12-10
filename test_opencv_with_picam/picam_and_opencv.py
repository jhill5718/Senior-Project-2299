import cv2
from picamera2 import Picamera2
import numpy as np
from libcamera import controls

height = 480
width = 640
middle = ((width//2),(height//2))
cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
cam.start()

while True:
	frame = cam.capture_array()
	cv2.circle(frame, middle, 10, (255, 0, 255), -1)
	cv2.imshow('f', frame)
	# cv2.waitKey(1) # if changed to 0, it updates every time you press a key
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
