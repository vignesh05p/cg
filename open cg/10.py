import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('content/cg.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur for smoothing
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Display the original and blurred images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred_img, cmap='gray')
plt.title("Blurred Image")
plt.axis("off")

plt.show()
