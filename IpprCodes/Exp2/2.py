import cv2
import numpy as np

def convert_to_24bit(image):
    # Ensure the image is in 24-bit format (BGR with 8 bits per channel)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def gray_negative_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Convert to 24-bit
    image_24bit = convert_to_24bit(image)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_24bit, cv2.COLOR_RGB2GRAY)
    
    # Compute the negative of the grayscale image
    negative_gray = 255 - gray_image
    
    # Save the negative grayscale image
    cv2.imwrite(output_path, negative_gray)
    print(f"Negative grayscale image saved at: {output_path}")

# Example usage
input_image_path = "urimg.jpg"  # Replace with actual image path
output_image_path = "negative_gray_output.jpg"
gray_negative_image(input_image_path, output_image_path)
