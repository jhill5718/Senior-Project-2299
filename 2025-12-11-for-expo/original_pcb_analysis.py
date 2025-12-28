import cv2
import matplotlib.pyplot as plt
import numpy as np

from picamera2 import Picamera2
from libcamera import controls

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

# --- Load images ---
img1 = cv2.imread("image1.png")
img2 = cv2.imread("image2.png")

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found. Make sure 'image1.png' and 'image2.png' are uploaded!")

# --- Resize to same dimensions ---
h = min(img1.shape[0], img2.shape[0])
w = min(img1.shape[1], img2.shape[1])
img1 = cv2.resize(img1, (w, h))
img2 = cv2.resize(img2, (w, h))

# --- Downscale → blur → upscale trick ---
def smooth_blur(img, scale=0.25, kernel=(15, 15), sigma=00):
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    blurred_small = cv2.GaussianBlur(small, kernel, sigma)
    return cv2.resize(blurred_small, (img.shape[1], img.shape[0]))

blur1 = smooth_blur(img1)
blur2 = smooth_blur(img2)

# --- Convert to grayscale for SSIM ---
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray_blur1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
gray_blur2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

# --- Compute SSIM difference maps ---
score_orig, diff_originals = ssim(gray1, gray2)
score_blur, diff_blurred = ssim(gray_blur1, gray_blur2)

# Invert SSIM to show differences
diff_originals = 1 - diff_originals
diff_blurred = 1 - diff_blurred

# Normalize for visualization
diff_originals = cv2.normalize(diff_originals, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
diff_blurred = cv2.normalize(diff_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print(f"SSIM (Originals): {score_orig:.4f}")
print(f"SSIM (Blurred):  {score_blur:.4f}")

# --- Display results ---
plt.figure(figsize=(10, 10))

# Row 1: Original images
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original Image 1")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Original Image 2")
plt.axis("off")

# Row 2: Blurred images
plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(blur1, cv2.COLOR_BGR2RGB))
plt.title("Blurred Image 1")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB))
plt.title("Blurred Image 2")
plt.axis("off")

# Row 3: Difference maps
plt.subplot(3, 2, 5)
plt.imshow(diff_originals, cmap='inferno')
plt.title("SSIM Difference Map (Originals)")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.imshow(diff_blurred, cmap='inferno')
plt.title("SSIM Difference Map (Blurred)")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- Optional save ---
cv2.imwrite("blurred_image1.png", blur1)
cv2.imwrite("blurred_image2.png", blur2)
cv2.imwrite("ssim_diff_originals.png", diff_originals)
cv2.imwrite("ssim_diff_blurred.png", diff_blurred)
