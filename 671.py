import cv2
import numpy as np

# Load image
image = cv2.imread(r"C:\Users\Darshil\Desktop\test_4_cutout.png")
if image is None:
    raise ValueError("Image not found or path is incorrect.")

# Resize for consistency
image = cv2.resize(image, (500, 500))

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define more lenient white color range in HSV
lower_white = np.array([0, 0, 180])     # Lowered V from 200 to 180
upper_white = np.array([180, 80, 255])  # Increased S from 50 to 80

# Create a mask for white-like regions
mask = cv2.inRange(hsv, lower_white, upper_white)

# Apply the mask to the image
result = cv2.bitwise_and(image, image, mask=mask)

# Find contours of white areas
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Calculate white coating percentage
white_area = cv2.countNonZero(mask)
total_area = image.shape[0] * image.shape[1]
white_percent = (white_area / total_area) * 100

print(f"Estimated white coating: {white_percent:.2f}%")

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('White Coating Mask', mask)
cv2.imshow('White Coating Detected', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
