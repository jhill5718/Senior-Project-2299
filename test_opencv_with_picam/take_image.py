import cv2
from picamera2 import Picamera2
import numpy as np
from libcamera import controls

height = 2592 #480
width = 4608 # 640
cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
cam.start()

while True:
	frame = cam.capture_array()
	# cv2.circle(frame, middle, 10, (255, 0, 255), -1)
	cv2.imshow('f', frame)
	# cv2.waitKey(1) # if changed to 0, it updates every time you press a key
	key = cv2.waitKey(1) &0xFF # &0xFF means only the last 8 bits
	
	if key == ord('c'):
		cv2.imwrite('image.jpg', frame)
		print('Regular image taken! SAVED AS: image.jpg')
	
	if key == ord('g'):
		gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imwrite("gray-image.jpg", gray1)
		print('Grayscale image taken! SAVED AS: gray-image.jpg')
		
	if key == ord('q'):
		break
