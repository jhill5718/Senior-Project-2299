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

# --- Pure-Python SSIM implementation ---
def ssim(img1, img2, window_size=11, K1=0.01, K2=0.03, L=255):
    """
    Compute the Structural Similarity Index (SSIM) between two grayscale images.
    Pure Python + NumPy + OpenCV, no C extensions.
    
    Returns:
        ssim_index : float, mean SSIM
        ssim_map   : 2D array of SSIM values
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    gauss = cv2.getGaussianKernel(window_size, 1.5)
    window = gauss @ gauss.T

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    ssim_index = np.mean(ssim_map)
    return ssim_index, ssim_map

while True:
	frame = cam.capture_array()
	cv2.circle(frame, middle, 10, (255, 0, 255), -1)
	cv2.imshow('f', frame)
	# cv2.waitKey(1) # if changed to 0, it updates every time you press a key
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
