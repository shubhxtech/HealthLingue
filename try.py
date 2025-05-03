import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, segmentation, color, filters
import pandas as pd

class TonguePapillaeAnalyzer:
    """
    Class for analyzing tongue images to detect papillae size and redness in local patches.
    """
    
    def __init__(self, patch_size=50, threshold_adjust=1.0, min_papillae_size=10, 
                 max_papillae_size=500, contrast_enhance=1.5):
        """
        Initialize the analyzer with parameters.
        
        Args:
            patch_size: Size of local patches for analysis
            threshold_adjust: Factor to adjust thresholding sensitivity
            min_papillae_size: Minimum size of papillae in pixels
            max_papillae_size: Maximum size of papillae in pixels
            contrast_enhance: Factor for contrast enhancement
        """
        self.patch_size = patch_size
        self.threshold_adjust = threshold_adjust
        self.min_papillae_size = min_papillae_size
        self.max_papillae_size = max_papillae_size
        self.contrast_enhance = contrast_enhance

    def preprocess_image(self, image):
        """
        Preprocess the tongue image for better analysis.
        
        Args:
            image: Input tongue image
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to LAB color space (better for redness detection)
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply contrast enhancement
        l, a, b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=self.contrast_enhance, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Reconstruct image
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_rgb, None, 10, 10, 7, 21)
        
        return denoised

    def segment_tongue(self, image):
        """
        Improved tongue segmentation with multiple strategies to ensure accurate isolation.
        
        Args:
            image: Input image
            
        Returns:
            Masked image containing only the tongue region
        """
        # Make a copy of the image
        img_copy = image.copy()
        
        # Convert to different color spaces for better segmentation
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
        ycrcb = cv2.cvtColor(img_copy, cv2.COLOR_RGB2YCrCb)
        
        # Extract color components that help identify tongue tissue
        s = hsv[:, :, 1]  # Saturation
        v = hsv[:, :, 2]  # Value (brightness)
        a = lab[:, :, 1]  # Green-Red component (higher for reddish tongue)
        cr = ycrcb[:, :, 1]  # Red component in YCrCb
        
        # Create a redness map (stronger signal for tongue tissue)
        # Normalize a-channel to [0, 255]
        a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cr_norm = cv2.normalize(cr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Combine multiple features for robust segmentation
        redness = cv2.addWeighted(a_norm, 0.5, cr_norm, 0.5, 0)
        
        # Apply bilateral filter to preserve edges while smoothing
        redness = cv2.bilateralFilter(redness, 9, 75, 75)
        
        # Use adaptive thresholding to handle varying lighting
        thresh1 = cv2.adaptiveThreshold(redness, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 25, -3)
        
        # Also try Otsu's method as a different approach
        _, thresh2 = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine both thresholding results
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 4)
        
        # If no components found, try a different approach
        if num_labels <= 1:
            # Fallback method: use color segmentation in HSV space
            # Define range for reddish tongue color
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red = np.array([160, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            
            # Combine masks
            tongue_mask = cv2.bitwise_or(mask1, mask2)
            tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_CLOSE, kernel)
            tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_OPEN, kernel)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tongue_mask, 4)
            if num_labels <= 1:
                # If still no components, return a minimal segmentation
                return image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        
        # Find the largest component by area (excluding background)
        max_area = 0
        max_label = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Also consider component position - tongue is usually central
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Distance from image center
            img_center_x = image.shape[1] // 2
            img_center_y = image.shape[0] // 2
            
            dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            center_factor = 1.0
            
            # Prefer central objects
            if dist_from_center < (image.shape[1] + image.shape[0]) / 4:
                center_factor = 1.5
                
            # Weighted area considering both size and position
            weighted_area = area * center_factor
            
            if weighted_area > max_area:
                max_area = weighted_area
                max_label = i
                
        # Create final tongue mask with only the largest component
        tongue_mask = np.zeros_like(labels, dtype=np.uint8)
        tongue_mask[labels == max_label] = 255
        
        # Final cleanup of the mask
        kernel = np.ones((9, 9), np.uint8)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_CLOSE, kernel)
        
        # Fill any small holes inside the mask
        contours, _ = cv2.findContours(tongue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(tongue_mask)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)
        
        # Apply the mask to the original image
        masked_image = image.copy()
        masked_image[filled_mask == 0] = 0
        
        return masked_image, filled_mask

    def detect_papillae_in_patch(self, patch):
        """
        Detect papillae in a local patch of the tongue.
        
        Args:
            patch: Local patch of the tongue image
            
        Returns:
            Labeled papillae, properties of detected papillae
        """
        # Convert to grayscale if RGB
        if len(patch.shape) == 3:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray_patch = patch
            
        # Use adaptive thresholding to identify potential papillae
        thresh = cv2.adaptiveThreshold(
            gray_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, self.threshold_adjust * 2
        )
        
        # Remove small noise
        cleaned = morphology.remove_small_objects(
            thresh.astype(bool), min_size=self.min_papillae_size
        )
        
        # Apply distance transform to separate close papillae
        dist_transform = ndimage.distance_transform_edt(cleaned)
        
        # Find local maxima
        local_max = filters.rank.maximum(dist_transform.astype(np.uint8), morphology.disk(2))
        markers = ndimage.label(local_max == dist_transform)[0]
        
        # Apply watershed segmentation
        segmented = segmentation.watershed(-dist_transform, markers, mask=cleaned)
        
        # Measure properties
        props = measure.regionprops(segmented, intensity_image=gray_patch)
        
        return segmented, props

    def analyze_redness(self, patch, segmented):
        """
        Analyze redness of detected papillae in a patch.
        
        Args:
            patch: Local patch of the tongue image
            segmented: Segmentation mask from the papillae detection
            
        Returns:
            Average redness values for each papillae
        """
        # Convert to LAB color space
        if len(patch.shape) == 3:
            lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        else:
            # If the patch is grayscale, convert to RGB first
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            lab_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2LAB)
            hsv_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2HSV)
        
        # Extract the a channel (red-green) and hue, saturation
        a_channel = lab_patch[:, :, 1]  # Higher values indicate more redness
        h_channel = hsv_patch[:, :, 0]  # Hue
        s_channel = hsv_patch[:, :, 1]  # Saturation
        
        # Calculate redness for each papillae
        redness_values = []
        
        unique_labels = np.unique(segmented)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        
        for label in unique_labels:
            # Create mask for this papilla
            papilla_mask = segmented == label
            
            # Measure average a-channel value (higher = more red)
            a_avg = np.mean(a_channel[papilla_mask])
            
            # Calculate normalized redness metric
            # (considering both a-channel and saturation)
            h_avg = np.mean(h_channel[papilla_mask])
            s_avg = np.mean(s_channel[papilla_mask])
            
            # Create a redness metric - higher values for more red papillae
            # Red is around 0-30 in hue space
            h_factor = 1.0
            if 0 <= h_avg <= 30 or 330 <= h_avg <= 360:
                h_factor = 2.0  # Boost redness score for red hues
                
            redness_score = (a_avg + 128) / 255 * s_avg / 255 * h_factor
            
            redness_values.append({
                'label': label,
                'a_channel': a_avg,
                'hue': h_avg,
                'saturation': s_avg,
                'redness_score': redness_score
            })
            
        return redness_values

    def extract_patches(self, image, mask):
        """
        Extract local patches from the tongue image for analysis.
        Only extracts patches that are fully within the tongue region.
        
        Args:
            image: Input tongue image
            mask: Tongue mask
            
        Returns:
            List of patches and their coordinates
        """
        patches = []
        coords = []
        
        height, width = mask.shape[:2]
        
        # Find bounding box of the tongue to focus only on relevant areas
        non_zero_points = cv2.findNonZero(mask)
        if non_zero_points is None:
            return [], []  # No tongue detected
            
        x, y, w, h = cv2.boundingRect(non_zero_points)
        
        # Only analyze within the bounding box (with small margin)
        margin = 5
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(width, x + w + margin)
        y_end = min(height, y + h + margin)
        
        # Extract patches with overlap
        for y_pos in range(y_start, y_end - self.patch_size + 1, self.patch_size // 2):
            for x_pos in range(x_start, x_end - self.patch_size + 1, self.patch_size // 2):
                patch = image[y_pos:y_pos+self.patch_size, x_pos:x_pos+self.patch_size]
                patch_mask = mask[y_pos:y_pos+self.patch_size, x_pos:x_pos+self.patch_size]
                
                # Only include patches that are mostly tongue (at least 70% of the patch)
                mask_coverage = np.sum(patch_mask > 0) / (self.patch_size * self.patch_size)
                if mask_coverage > 0.7:
                    patches.append(patch)
                    coords.append((x_pos, y_pos))
        
        return patches, coords

    def analyze_image(self, image_path):
        """
        Main method to analyze a tongue image with improved segmentation visualization.
        
        Args:
            image_path: Path to the tongue image
            
        Returns:
            DataFrame with papillae measurements and visualization images
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if image is too large
        max_dim = 800
        if max(image.shape[:2]) > max_dim:
            scale = max_dim / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Segment tongue
        segmented_tongue, tongue_mask = self.segment_tongue(preprocessed)
        
        # Create a visualization of the tongue segmentation with contour outline
        # instead of a full overlay to better see the segmentation boundary
        segmentation_viz = image.copy()
        
        # Find contours of the mask
        contours, _ = cv2.findContours(tongue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contour on the original image in green with 2px thickness
        cv2.drawContours(segmentation_viz, contours, -1, (0, 255, 0), 2)
        
        # Extract patches
        patches, coords = self.extract_patches(segmented_tongue, tongue_mask)
        
        # Process each patch
        results = []
        visualization = image.copy()
        
        # Create a separate patch visualization
        patch_viz = image.copy()
        
        for i, (patch, (x, y)) in enumerate(zip(patches, coords)):
            # Detect papillae
            segmented, props = self.detect_papillae_in_patch(patch)
            
            if len(props) == 0:
                continue
                
            # Analyze redness
            redness_data = self.analyze_redness(patch, segmented)
            
            # Store results
            for j, prop in enumerate(props):
                if self.min_papillae_size <= prop.area <= self.max_papillae_size:
                    # Find corresponding redness data
                    red_data = next((r for r in redness_data if r['label'] == prop.label), None)
                    
                    if red_data:
                        results.append({
                            'patch_id': i,
                            'papilla_id': j,
                            'x': x + prop.centroid[1],
                            'y': y + prop.centroid[0],
                            'size': prop.area,
                            'perimeter': prop.perimeter,
                            'eccentricity': prop.eccentricity,
                            'redness_score': red_data['redness_score'],
                            'a_channel': red_data['a_channel'],
                            'hue': red_data['hue'],
                            'saturation': red_data['saturation']
                        })
            
            # Visualize this patch on the patch visualization with a thin border
            cv2.rectangle(patch_viz, (x, y), (x+self.patch_size, y+self.patch_size), (0, 255, 0), 1)
        
        # Create dataframe
        df = pd.DataFrame(results)
        
        return df, visualization, segmentation_viz, patch_viz

    def visualize_results(self, image, df):
        """
        Visualize the detected papillae and their properties.
        
        Args:
            image: Original image
            df: DataFrame with papillae measurements
            
        Returns:
            Visualization image
        """
        # Create a copy for visualization
        viz_image = image.copy()
        
        # Define color map for redness (blue to red)
        if len(df) > 0:
            min_redness = df['redness_score'].min()
            max_redness = df['redness_score'].max()
            redness_range = max_redness - min_redness
            
            # Draw circles for each papilla
            for _, row in df.iterrows():
                x, y = int(row['x']), int(row['y'])
                size = int(np.sqrt(row['size'] / np.pi))
                
                # Normalize redness score
                if redness_range > 0:
                    norm_redness = (row['redness_score'] - min_redness) / redness_range
                else:
                    norm_redness = 0.5
                
                # Create color (blue to red)
                color = (int(255 * (1 - norm_redness)), 0, int(255 * norm_redness))
                
                # Draw the papilla
                cv2.circle(viz_image, (x, y), max(1, size), color, 1)
        
        return viz_image

    def generate_report(self, df):
        """
        Generate a statistical report of the papillae analysis.
        
        Args:
            df: DataFrame with papillae measurements
            
        Returns:
            Dictionary with statistical metrics
        """
        if len(df) == 0:
            return {
                'total_papillae': 0,
                'avg_size': 0,
                'avg_redness': 0
            }
            
        # Calculate statistics
        total_papillae = len(df)
        avg_size = df['size'].mean()
        avg_redness = df['redness_score'].mean()
        
        # Group by patch
        patch_stats = df.groupby('patch_id').agg({
            'size': 'mean',
            'redness_score': 'mean',
            'papilla_id': 'count'
        }).rename(columns={'papilla_id': 'count'})
        
        # Find patches with highest redness
        high_redness_patches = patch_stats.sort_values('redness_score', ascending=False).head(3)
        
        # Prepare report
        report = {
            'total_papillae': total_papillae,
            'avg_size': avg_size,
            'avg_redness': avg_redness,
            'patch_density': patch_stats['count'].mean(),
            'high_redness_patches': high_redness_patches.index.tolist(),
            'high_redness_values': high_redness_patches['redness_score'].tolist()
        }
        
        return report

# Example usage
def main():
    # Create analyzer instance
    analyzer = TonguePapillaeAnalyzer(
        patch_size=64, 
        threshold_adjust=1.2,
        min_papillae_size=15,
        max_papillae_size=400
    )
    
    # Path to tongue image
    image_path = "1.jpeg"
    
    try:
        # Analyze image
        results_df, visualization_image, segmentation_viz, patch_viz = analyzer.analyze_image(image_path)
        
        # Visualize results
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualization = analyzer.visualize_results(image, results_df)
        
        # Generate report
        report = analyzer.generate_report(results_df)
        
        # Display results
        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title("Tongue Segmentation")
        plt.imshow(segmentation_viz)
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Detected Papillae")
        plt.imshow(visualization)
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title("Papillae Size vs Redness")
        if len(results_df) > 0:
            plt.scatter(
                results_df['size'], 
                results_df['redness_score'], 
                c=results_df['redness_score'], 
                cmap='coolwarm',
                alpha=0.7
            )
            plt.colorbar(label='Redness Score')
            plt.xlabel('Papilla Size (pixels)')
            plt.ylabel('Redness Score')
        else:
            plt.text(0.5, 0.5, "No papillae detected", ha='center', va='center')
            
        plt.tight_layout()
        
        # Save visualization results
        plt.savefig("papillae_analysis_result.png")
        
        # Create a second figure to show patch-based analysis
        if len(results_df) > 0:
            plt.figure(figsize=(12, 10))
            plt.title("Tongue Patches for Analysis (Only within tongue region)")
            plt.imshow(patch_viz)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("tongue_patches.png")
        
        plt.show()
        
        # Print report
        print("\nAnalysis Report:")
        print(f"Total papillae detected: {report['total_papillae']}")
        print(f"Average papilla size: {report['avg_size']:.2f} pixels")
        print(f"Average redness score: {report['avg_redness']:.2f}")
        print(f"Average papillae per patch: {report['patch_density']:.2f}")
        
        # Save results to CSV
        if len(results_df) > 0:
            results_df.to_csv("papillae_measurements.csv", index=False)
            print("\nMeasurements saved to 'papillae_measurements.csv'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()