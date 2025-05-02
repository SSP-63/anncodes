import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Avoids GTK-related crashes
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.jpeg', 0)

# Check if image is loaded
if img is None:
    print("Image not found. Please check the filename and path.")
    exit()

# Threshold the image to binary (important for morphology)
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Define kernel (structuring element)
kernel = np.ones((5, 5), np.uint8)

# Morphological operations
erosion = cv2.erode(binary_img, kernel, iterations=1)
dilation = cv2.dilate(binary_img, kernel, iterations=1)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# Plot all results using matplotlib
titles = ['Original', 'Binary', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [img, binary_img, erosion, dilation, opening, closing]

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
