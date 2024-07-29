import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('content/cg.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection using Canny
edges = cv2.Canny(gray, 100, 200)

# Texture extraction using averaging filter
kernel = np.ones((5, 5), np.float32) / 25
texture = cv2.filter2D(gray, -1, kernel)

# Display the original image, edges, and texture
plt.figure(figsize=(15, 5))
titles = ['Original Image', 'Edges', 'Texture']
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), edges, texture]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray' if i > 0 else None)
    plt.axis('off')

plt.show()
