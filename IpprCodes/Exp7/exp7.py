# Load image

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use 'Agg' if you only want to save plots

import matplotlib.pyplot as plt

img = cv2.imread('image.jpeg', 0)

# 1. Sobel Edge Detection
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)

# 2. Prewitt Edge Detection (by custom kernel)
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittx = cv2.filter2D(img, -1, kernelx)
prewitty = cv2.filter2D(img, -1, kernely)
prewitt_combined = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)

# 3. Canny Edge Detection
canny = cv2.Canny(img, 100, 200)

# Display
titles = ['Original', 'Sobel', 'Prewitt', 'Canny']
images = [img, sobel_combined, prewitt_combined, canny]

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
