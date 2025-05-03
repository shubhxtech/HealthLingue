import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize image
image = cv2.imread(r"tongue3.png")
if image is None:
    raise ValueError("Image not found or path is incorrect.")
image = cv2.resize(image, (500, 500))

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Define thresholds for white coating
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 80, 255])
brightness_threshold = 215

# Filter white-like coating
white_like_mask = ((h >= lower_white[0]) & (h <= upper_white[0]) &
                   (s >= lower_white[1]) & (s <= upper_white[1]) &
                   (v >= lower_white[2]) & (v <= upper_white[2]))
filtered_white_mask = white_like_mask & (v > brightness_threshold)

# Create highlight mask
final_mask = np.zeros_like(v, dtype=np.uint8)
final_mask[filtered_white_mask] = 255

# Create visuals
highlighted = cv2.bitwise_and(image, image, mask=final_mask)

# Brightness map (colormap on V)
v_normalized = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
v_colormap = cv2.applyColorMap(v_normalized, cv2.COLORMAP_JET)

# White coating heatmap (coating intensity only)
heatmap = np.zeros_like(image)
heatmap[filtered_white_mask] = cv2.applyColorMap(v, cv2.COLORMAP_HOT)[filtered_white_mask]

# Convert BGR to RGB for matplotlib display
highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
v_colormap_rgb = cv2.cvtColor(v_colormap, cv2.COLOR_BGR2RGB)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ---- Subplot Display ----
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(v_colormap_rgb)
plt.title('Brightness Map (V Channel)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(highlighted_rgb)
plt.title('Filtered White Highlight')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(heatmap_rgb)
plt.title('White Coating Heatmap')
plt.axis('off')

plt.tight_layout()
plt.show()
