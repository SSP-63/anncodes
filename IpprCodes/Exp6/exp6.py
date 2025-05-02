import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('image.jpeg', 0)

# 1. Gaussian Smoothing (Spatial Domain)
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

# 2. Median Filtering (Good for salt & pepper noise)
median_blur = cv2.medianBlur(img, 5)

# 3. Sharpening Kernel
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, kernel_sharpen)

# 4. Frequency Domain Filter (Low-pass Filter)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# Display
titles = ['Original', 'Gaussian Blur', 'Median Blur', 'Sharpened', 'Low-pass Filter (Freq)']
images = [img, gaussian_blur, median_blur, sharpened, img_back]

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
