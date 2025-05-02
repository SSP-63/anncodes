import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend to avoid tkinter issues
import matplotlib.pyplot as plt

# Load the input image
image = cv2.imread("urimg.jpg")  # Replace with your image file
if image is None:
    raise FileNotFoundError("Error: Image not found. Please check the file path.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
h, w = image.shape[:2]  # Get image dimensions

# 1) Translation: Move the image 50 pixels right and 30 pixels down
translation_matrix = np.float32([[1, 0, 1000], [0, 1, 500]])
translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

# 2) Rotation: Rotate image 45 degrees counterclockwise around the center
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# 3) Scaling: Resize image to 1.5x along x and 0.8x along y
scaled_image = cv2.resize(image, None, fx=1.5, fy=0.8)

# 4) Shearing: Apply a shear transformation along the x-axis
shear_matrix = np.float32([[1, 0.5, 0], [0, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (w, h))

# Display all images in a 2x3 grid and save the result
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Show original and transformed images
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(translated_image)
axes[0, 1].set_title("Translated Image")
axes[0, 2].imshow(rotated_image)
axes[0, 2].set_title("Rotated Image")

axes[1, 0].imshow(scaled_image)
axes[1, 0].set_title("Scaled Image")
axes[1, 1].imshow(sheared_image)
axes[1, 1].set_title("Sheared Image")

# Hide the last empty subplot
axes[1, 2].axis('off')

# Adjust layout and save instead of showing (due to tkinter issue)
plt.tight_layout()
plt.savefig("output_transformations.png")  # Save output as an image
print("✅ Output saved as 'output_transformations.png'. Open it to view results.")