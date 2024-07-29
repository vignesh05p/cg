import cv2
import numpy as np
import matplotlib.pyplot as plt

def translate_image(image, dx, dy):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

# Read the image
image = cv2.imread(r'content\cg.jpg')

# Get image dimensions
height, width = image.shape[:2]

# Calculate the center coordinates of the image
center = (width // 2, height // 2)

# Get user inputs
rotation_value = int(input("Enter the degree of Rotation: "))
scaling_value = float(input("Enter the zooming factor: "))
dx = int(input("Enter the horizontal translation (pixels): "))
dy = int(input("Enter the vertical translation (pixels): "))

# Create the 2D rotation matrix and apply rotation
rotation_matrix = cv2.getRotationMatrix2D(center, rotation_value, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Create the scaling matrix and apply scaling
scaling_matrix = cv2.getRotationMatrix2D(center, 0, scaling_value)
scaled_image = cv2.warpAffine(rotated_image, scaling_matrix, (width, height))

# Apply translation
translated_image = translate_image(scaled_image, dx, dy)

# Save the final image
cv2.imwrite('Final_image.png', translated_image)

# Display the original and final images using Matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Transformed Image')
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
