import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('content/cg.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(10, 5))
plt.title('Contours on Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
