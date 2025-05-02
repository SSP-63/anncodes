import cv2
import numpy as np

def negative_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Compute the negative image
    negative = 255 - image
    
    # Save the negative image
    cv2.imwrite(output_path, negative)
    print(f"Negative image saved at: {output_path}")

# Example usage
input_image_path = "urimg.jpg"  # Replace with actual image path
output_image_path = "negative_output.jpg"
negative_image(input_image_path, output_image_path)
