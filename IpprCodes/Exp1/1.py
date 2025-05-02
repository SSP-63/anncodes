from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = Image.open("images.jpg")
img = img.convert("RGB")

# Convert the image to a NumPy array for easier manipulation
img_array = np.array(img)

# Split the RGB channels
r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

# Create images for the individual channels
r_image = np.zeros_like(img_array)
r_image[:,:,0] = r  # Only keep the red channel

g_image = np.zeros_like(img_array)
g_image[:,:,1] = g  # Only keep the green channel

b_image = np.zeros_like(img_array)
b_image[:,:,2] = b  # Only keep the blue channel

# Convert to CMY channels (1 - RGB)
c_image = 255 - r_image
m_image = 255 - g_image
y_image = 255 - b_image

# Plot the images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Original image
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# RGB Channels
axes[0, 1].imshow(r_image)
axes[0, 1].set_title("Red Channel")
axes[0, 1].axis('off')

axes[0, 2].imshow(g_image)
axes[0, 2].set_title("Green Channel")
axes[0, 2].axis('off')

axes[0, 3].imshow(b_image)
axes[0, 3].set_title("Blue Channel")
axes[0, 3].axis('off')

# CMY Channels
axes[1, 0].imshow(c_image)
axes[1, 0].set_title("Cyan Channel")
axes[1, 0].axis('off')

axes[1, 1].imshow(m_image)
axes[1, 1].set_title("Magenta Channel")
axes[1, 1].axis('off')

axes[1, 2].imshow(y_image)
axes[1, 2].set_title("Yellow Channel")
axes[1, 2].axis('off')

# Hide the empty subplot
axes[1, 3].axis('off')

# Display the plots
plt.tight_layout()
plt.show()
