import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import opening, closing, disk, remove_small_holes
from skimage.filters import threshold_otsu
import requests  # Import requests for fetching Jaggedness from the server

# Function to get Jaggedness from FastAPI server
def get_jaggedness():
    try:
        # URL of the FastAPI server where the /jaggedness endpoint is exposed
        response = requests.get("http://127.0.0.1:5000/jaggedness")  # Update this URL if your server is running elsewhere

        if response.status_code == 200:
            data = response.json()
            jaggedness = data.get("Jaggedness", None)  # Retrieve the value from the JSON response
            if jaggedness is not None:
                print(f"Jaggedness Value: {jaggedness}")
                return jaggedness
            else:
                print("Jaggedness value not found in response.")
                return None
        else:
            print(f"Failed to retrieve jaggedness. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while fetching jaggedness: {str(e)}")
        return None

# Graham Scan Algorithm to compute the convex hull
def graham_scan(points):
    points = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])

def compute_and_plot_major_dents_concavity(image_path, r1=3, r2=3, A_max=64, sigma=18, sigma_green=3, concavity_thresh=-0.015, length_thresh=10):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val = threshold_otsu(gray)
    mask = (gray > thresh_val).astype(np.uint8)

    mask_clean = opening(mask.astype(bool), disk(r1))
    mask_clean = closing(mask_clean, disk(r2)).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_clean)
    if num_labels > 1:
        largest = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
        mask_clean = (labels == largest).astype(np.uint8)
    mask_filled = remove_small_holes(mask_clean.astype(bool), area_threshold=A_max).astype(np.uint8)

    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).squeeze()
    x_vals, y_vals = contour[:, 0], contour[:, 1]

    y_thresh = np.min(y_vals) + 0.2 * (np.max(y_vals) - np.min(y_vals))
    upper_indices = np.where(y_vals <= y_thresh)[0]
    top_left = contour[upper_indices[np.argmin(x_vals[upper_indices])]]
    top_right = contour[upper_indices[np.argmax(x_vals[upper_indices])]]
    bottom_y = np.max(y_vals)
    bottom_indices = np.where(y_vals == bottom_y)[0]
    bottom = contour[bottom_indices[np.argmin(np.abs(x_vals[bottom_indices] - np.median(x_vals)))]]


    def find_index(pt): return np.where((contour == pt).all(axis=1))[0][0]
    i_tl = find_index(top_left)
    i_tr = find_index(top_right)
    i_b = find_index(bottom)

    def arc_indices(start, end, N):
        return np.arange(start, end + 1) if start <= end else np.concatenate([np.arange(start, N), np.arange(0, end + 1)])

    N = len(contour)
    path1 = arc_indices(i_tl, i_tr, N)
    path2 = arc_indices(i_tr, i_tl, N)
    seg_idx = path1 if i_b in path1 else path2

    # Get the original red segment
    segment = contour[seg_idx]
    xs, ys = segment[:, 0].astype(float), segment[:, 1].astype(float)

    # Apply smoothing to the red segment (original segment) with sigma=18
    xs_smooth = gaussian_filter1d(xs, sigma)
    ys_smooth = gaussian_filter1d(ys, sigma)

    # Create a green segment with sigma=5 (smoother version)
    xs_green = gaussian_filter1d(xs, sigma_green)
    ys_green = gaussian_filter1d(ys, sigma_green)

    # Calculate first and second derivatives (concavity)
    dx = np.gradient(xs_green)
    dy = np.gradient(ys_green)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Concavity thresholding: Only consider regions with a highly negative second derivative
    concavity = d2x + d2y  # Combined second derivative for concavity

    # Count significant dents based on concavity
    major_dent_count = 0
    is_in_dent = False  # Flag to track if we are in a significant dent
    dent_length = 0  # Track the length of continuous concave regions
    last_dent_index = -1  # To store the index of the last dent

    for i in range(1, len(xs_green) - 1):
        if concavity[i] < concavity_thresh:
            if not is_in_dent:
                dent_length = 1  # Start a new dent region
                is_in_dent = True
            else:
                dent_length += 1  # Continue the dent region
            last_dent_index = i
        else:
            if is_in_dent and dent_length >= length_thresh:
                major_dent_count += 1  # Count it as a major dent if its length exceeds threshold
            is_in_dent = False  # Reset if not a dent

    # Function to calculate perimeter
    def perimeter(xc, yc):
        return np.sum(np.hypot(np.diff(xc, append=xc[0]), np.diff(yc, append=yc[0])))

    # Calculate perimeter of both red and blue segments
    P_red = perimeter(xs, ys)
    P_blue = perimeter(xs_smooth, ys_smooth)

    # Calculate the perimeter score
    perimeter_score = 1 - (P_blue / P_red)

    # Get Jaggedness value from the FastAPI server
    jaggedness = get_jaggedness()

    # If Jaggedness value is available, adjust the final score based on it
    if jaggedness is not None:
        final_jagged_score = (perimeter_score * 10) + major_dent_count + jaggedness
    else:
        final_jagged_score = (perimeter_score * 10) + major_dent_count

    # Commented out the plotting section
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask, cmap='gray')
    # plt.plot(xs, ys, 'r-', linewidth=2, label="Original Red Segment")  # Original red contour
    # plt.plot(xs_smooth, ys_smooth, 'b-', linestyle='--', linewidth=2, label="Smoothed Red Segment")  # Smoothed red contour
    # plt.plot(xs_green, ys_green, 'g-', linestyle='-', linewidth=2, label="Green Smoothed Segment (sigma=5)")  # Green contour
    #
    # # Display top-left, top-right, and bottom points
    # plt.plot(top_left[0], top_left[1], 'bo')
    # plt.text(top_left[0] + 3, top_left[1], "Top-Left", color='blue', fontsize=9)
    # plt.plot(top_right[0], top_right[1], 'bo')
    # plt.text(top_right[0] + 3, top_right[1], "Top-Right", color='blue', fontsize=9)
    # plt.plot(bottom[0], bottom[1], 'bo')
    # plt.text(bottom[0] + 3, bottom[1], "Bottom", color='blue', fontsize=9)
    #
    # plt.title(f"Contour with Major Dents (Dent Count: {major_dent_count})")
    # plt.axis("equal")
    # plt.legend()
    # plt.show()

    return {
        'major_dent_count': major_dent_count,
        'perimeter_score': perimeter_score,
        'final_jagged_score': final_jagged_score,  # Return final jagged score
        'segment_contour': segment,
        'smoothed_segment': np.vstack((xs_smooth, ys_smooth)).T,
        'green_segment': np.vstack((xs_green, ys_green)).T
    }

# Run the final version with concavity-based dent detection on the green segment
result = compute_and_plot_major_dents_concavity("tongue_output.jpg")
print("Final Jagged score:", result['final_jagged_score'])
