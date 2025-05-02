import cv2
import numpy as np

# Load the image from directory
image_path = "urimg.jpg"  # Change this to your image file
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found. Check the path.")
    exit()

rows, cols, _ = img.shape

# Translation
M_translation = np.float32([[1, 0, 50], [0, 1, 50]])  # Shift right by 50, down by 50
img_translated = cv2.warpAffine(img, M_translation, (cols, rows))

# Rotation (90 degrees counterclockwise)
M_rotation = cv2.getRotationMatrix2D((cols//2, rows//2), 90, 1)
img_rotated = cv2.warpAffine(img, M_rotation, (cols, rows))

# Scaling
img_scaled = cv2.resize(img, (cols, rows))  # Resize back to original size for consistency

# Shearing (x-axis shear)
shear_factor = 0.5
M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
img_sheared = cv2.warpAffine(img, M_shear, (cols, rows))

# Create a single frame to show all images together
top_row = np.hstack([img, img_translated])
bottom_row = np.hstack([img_rotated, img_sheared])
final_frame = np.vstack([top_row, bottom_row])

# Save the final image
output_path = "transformed_image.jpg"
cv2.imwrite(output_path, final_frame)

# Display the final collage
cv2.imshow("Transformations", final_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Final transformed image saved as {output_path}")
