"""
Script to verify that inner and outer boundaries are correctly calculated.

This script visualizes the border, inner boundary, and outer boundary
to ensure they are properly oriented.
"""

import sys
from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cvimage import CVImage


def visualize_boundaries():
    """Visualize the embryo border and its inner/outer extensions."""
    print("=" * 70)
    print("BOUNDARY VERIFICATION")
    print("=" * 70)
    
    # Load a sample image
    data_path = Path('/Volumes/X2/Projects/phd/staging/data/data')
    folders = [f for f in data_path.glob('*') if f.is_dir()]
    
    if not folders:
        print("No data folders found!")
        return
    
    first_folder = folders[0]
    images = list(first_folder.glob('*.png'))
    
    if not images:
        print(f"No images found in {first_folder}")
        return
    
    # Load an image
    img_idx = min(10, len(images) - 1)
    img_path = images[img_idx]
    I = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    print(f"\nLoaded image: {img_path.name}")
    
    # Create CVImage instance
    embryo = CVImage(I, id=1.0, size=(512, 512), padding=44)
    
    # Process to get boundaries
    embryo.preprocess()
    embryo.segment_embryo_image()
    embryo.border_finder()
    embryo.extend_border()
    
    print(f"\nBoundary Parameters:")
    print(f"  inward:  {embryo.inward} pixels (positive = extends INWARD)")
    print(f"  outward: {embryo.outward} pixels (negative = extends OUTWARD)")
    print(f"  depth:   {embryo.depthInImage} pixels total")
    
    # Get the preprocessed image for visualization
    img_vis = embryo.get_image('preprocessed')
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original border
    plt.subplot(1, 3, 1)
    plt.imshow(img_vis, cmap='gray')
    plt.plot(embryo.x, embryo.y, 'b-', linewidth=2, label='Original Border')
    plt.title('Original Embryo Border')
    plt.legend()
    plt.axis('off')
    
    # Plot 2: All three boundaries
    plt.subplot(1, 3, 2)
    plt.imshow(img_vis, cmap='gray', alpha=0.5)
    plt.plot(embryo.x, embryo.y, 'b-', linewidth=2, label='Original Border')
    plt.plot(embryo.xext, embryo.yext, 'r-', linewidth=2, label=f'Outer (xext, yext)\noutward={embryo.outward}px')
    plt.plot(embryo.xint, embryo.yint, 'g-', linewidth=2, label=f'Inner (xint, yint)\ninward={embryo.inward}px')
    plt.title('All Boundaries')
    plt.legend()
    plt.axis('off')
    
    # Plot 3: Zoomed section to see detail
    plt.subplot(1, 3, 3)
    # Pick a section of the contour to zoom in
    center_x = int(np.mean(embryo.x))
    center_y = int(np.mean(embryo.y))
    zoom_size = 100
    
    x_min, x_max = center_x - zoom_size, center_x + zoom_size
    y_min, y_max = center_y - zoom_size, center_y + zoom_size
    
    plt.imshow(img_vis[y_min:y_max, x_min:x_max], cmap='gray', alpha=0.5,
               extent=[x_min, x_max, y_max, y_min])
    
    # Find points in the zoom region
    mask = ((embryo.x >= x_min) & (embryo.x <= x_max) &
            (embryo.y >= y_min) & (embryo.y <= y_max))
    
    if np.any(mask):
        plt.plot(embryo.x[mask], embryo.y[mask], 'bo-', linewidth=2, markersize=4, label='Original')
        plt.plot(embryo.xext[mask], embryo.yext[mask], 'ro-', linewidth=2, markersize=4, label='Outer (xext)')
        plt.plot(embryo.xint[mask], embryo.yint[mask], 'go-', linewidth=2, markersize=4, label='Inner (xint)')
    
    plt.title('Zoomed Detail')
    plt.legend()
    plt.axis('on')
    
    plt.tight_layout()
    plt.savefig('/tmp/boundary_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: /tmp/boundary_verification.png")
    plt.close()
    
    # Verify mathematically
    print(f"\nMathematical Verification:")
    print(f"-" * 70)
    
    # Calculate distances from original border
    dist_to_outer = np.sqrt((embryo.xext - embryo.x)**2 + (embryo.yext - embryo.y)**2)
    dist_to_inner = np.sqrt((embryo.xint - embryo.x)**2 + (embryo.yint - embryo.y)**2)
    
    print(f"Distance from original to outer (xext, yext):")
    print(f"  Mean: {np.mean(dist_to_outer):.2f} px (expected: {abs(embryo.outward):.2f} px)")
    print(f"  Std:  {np.std(dist_to_outer):.2f} px")
    
    print(f"\nDistance from original to inner (xint, yint):")
    print(f"  Mean: {np.mean(dist_to_inner):.2f} px (expected: {abs(embryo.inward):.2f} px)")
    print(f"  Std:  {np.std(dist_to_inner):.2f} px")
    
    # Check which boundary is actually inside vs outside
    print(f"\nOrientation Check:")
    print(f"-" * 70)
    
    # Calculate center of mass of the border
    center_x = np.mean(embryo.x)
    center_y = np.mean(embryo.y)
    
    # Calculate average distance from center for each boundary
    dist_orig_from_center = np.mean(np.sqrt((embryo.x - center_x)**2 + (embryo.y - center_y)**2))
    dist_outer_from_center = np.mean(np.sqrt((embryo.xext - center_x)**2 + (embryo.yext - center_y)**2))
    dist_inner_from_center = np.mean(np.sqrt((embryo.xint - center_x)**2 + (embryo.yint - center_y)**2))
    
    print(f"Average distance from center of embryo:")
    print(f"  Original border: {dist_orig_from_center:.2f} px")
    print(f"  Outer (xext):    {dist_outer_from_center:.2f} px")
    print(f"  Inner (xint):    {dist_inner_from_center:.2f} px")
    
    print(f"\nExpected ordering (from center outward):")
    print(f"  Inner (xint) < Original < Outer (xext)")
    
    print(f"\nActual ordering (from center outward):")
    boundaries = [
        ('Inner (xint)', dist_inner_from_center),
        ('Original', dist_orig_from_center),
        ('Outer (xext)', dist_outer_from_center)
    ]
    sorted_boundaries = sorted(boundaries, key=lambda x: x[1])
    for i, (name, dist) in enumerate(sorted_boundaries):
        print(f"  {i+1}. {name:15s} ({dist:.2f} px)")
    
    # Final verdict
    print(f"\n" + "=" * 70)
    if (dist_inner_from_center < dist_orig_from_center < dist_outer_from_center):
        print("✓ CORRECT: Inner is inside, Outer is outside")
        print("  - xint, yint (inward=40) are INNER boundaries")
        print("  - xext, yext (outward=-24) are OUTER boundaries")
    else:
        print("✗ INCORRECT: Boundaries are reversed!")
        print("  - The naming or calculations need to be fixed")
        if dist_outer_from_center < dist_inner_from_center:
            print("  - PROBLEM: xext (outer) is actually inside xint (inner)")
            print("  - SOLUTION: Swap the variable names or negate the lengths")
    print("=" * 70)


if __name__ == "__main__":
    visualize_boundaries()
