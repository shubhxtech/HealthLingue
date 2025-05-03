import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def detect_tongue_cracks_advanced(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    orig = img.copy()
    step_images = {}

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step_images['Grayscale'] = gray

    # Step 2: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    step_images['CLAHE'] = enhanced

    # # Step 3: Gaussian Blur
    # blurred = cv2.GaussianBlur(enhanced, (0,0), 0)
    # step_images['Blurred'] = blurred

    # Step 4: Morphological Closing (connect gaps in cracks)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=2)
    step_images['Morph Closed'] = morph

    # Step 5: Edge Detection
    edges = cv2.Canny(morph, 30, 100)
    step_images['Canny Edges'] = edges

    # Step 6: Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=20, maxLineGap=5)

    crack_img = orig.copy()
    total_length = 0
    valid_cracks = 0
    longest = 0

    angles = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.degrees(np.arctan2(dy, dx))

            # Filter nearly horizontal lines (not likely to be cracks)
            if abs(angle) < 20 or abs(angle) > 160:
                continue

            # Filter out small noise lines
            if length < 20:
                continue

            # Valid crack
            vertical_lines.append(((x1, y1), (x2, y2)))
            total_length += length
            longest = max(longest, length)
            valid_cracks += 1
            angles.append(angle)
            cv2.line(crack_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Step 7: Scoring from 0 to 10 (normalized)
    max_cracks = 800
    max_length = 25000
    crack_score_weight = 0.6
    length_score_weight = 0.4

    alpha = 0.8   # weight for number of cracks
    beta = 0.0015  # weight for total length

# Normalize each part to 0–1
    crack_score = min(valid_cracks, max_cracks) / max_cracks
    length_score = min(total_length, max_length) / max_length

    # Weighted sum
    score_raw = (crack_score_weight * crack_score +
                length_score_weight * length_score)

    # Scale to 0–10
    score = round(score_raw * 10, 2)



    # Convert for matplotlib
    step_images['Final Result'] = cv2.cvtColor(crack_img, cv2.COLOR_BGR2RGB)

    crack_info = {
        "crack_count": valid_cracks,
        "total_length": round(total_length, 2),
        "longest_crack": round(longest, 2),
        "score": score,
        "avg_angle": round(np.mean(angles), 2) if angles else None
    }
    
    return step_images['Final Result'], crack_info['score']


def show_processing_steps(step_images, crack_info):
    n = len(step_images)
    plt.figure(figsize=(18, 4))
    for i, (title, image) in enumerate(step_images.items()):
        plt.subplot(1, n, i + 1)
        cmap = 'gray' if len(image.shape) == 2 else None
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.suptitle(f"Score: {crack_info['score']} | Cracks: {crack_info['crack_count']} | Total Length: {crack_info['total_length']} px", fontsize=14)
    plt.tight_layout()
    plt.show()


# MAIN USAGE
if __name__ == "__main__":
    for i in range(8, 9):
        # image_path = f"test{i}.jpg"
        image_path = f"test5.jpg"  # Replace with your tongue image path
        print(f"Processing {image_path}...")
        # image_path = "test1.jpg"  # Replace with your tongue image path
    # image_path = "test2.jpg"  # Replace with your tongue image path
        steps, info = detect_tongue_cracks_advanced(image_path)
        print("Crack Analysis:", info)
        show_processing_steps(steps, info)
