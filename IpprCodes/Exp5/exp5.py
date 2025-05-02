import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Safe backend for Linux systems
import matplotlib.pyplot as plt
import pywt  # For DWT

# Load image in grayscale
img = cv2.imread('image.jpeg', 0)
if img is None:
    print("Image not found!")
    exit()

# ------------------- DFT -------------------
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Shift zero freq to center
magnitude_dft = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_dft = 20 * np.log(1 + magnitude_dft)

# ------------------- DCT -------------------
img_resized = cv2.resize(img, (256, 256))  # Resize for DCT
dct = cv2.dct(np.float32(img_resized))
magnitude_dct = np.log(abs(dct) + 1)

# ------------------- DWT -------------------
coeffs2 = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs2  # Approximation and detail coefficients

# ------------------- Display -------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(magnitude_dft, cmap='gray')
plt.title("DFT Magnitude Spectrum")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(magnitude_dct, cmap='gray')
plt.title("DCT Coefficients")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cA, cmap='gray')
plt.title("DWT Approximation (cA)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cH, cmap='gray')
plt.title("DWT Horizontal (cH)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cV, cmap='gray')
plt.title("DWT Vertical (cV)")
plt.axis('off')

plt.tight_layout()
plt.show()
