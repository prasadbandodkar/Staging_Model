"""
Integration test to verify the complete unroll pipeline works correctly.

This test creates a synthetic circular embryo image and verifies that:
1. The image is segmented correctly
2. Border points are found and extended
3. The unroll process completes without errors
4. The output has the expected dimensions
"""

import numpy as np
import cv2 as cv
import sys
sys.path.insert(0, 'src')

from src.cvimage import CVImage


def create_synthetic_embryo(size=600, ring_radius=200, ring_thickness=60):
    """
    Create a synthetic embryo image with a nuclear layer ring.
    
    Args:
        size: Image size (square)
        ring_radius: Radius of the nuclear layer center
        ring_thickness: Thickness of the nuclear layer
    
    Returns:
        Synthetic embryo image as uint8 grayscale
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    # Draw the embryo (filled circle)
    cv.circle(img, center, ring_radius + ring_thickness, 200, -1)
    
    # Add some texture to the nuclear layer
    for angle in range(0, 360, 10):
        rad = np.radians(angle)
        for r in range(ring_radius - 10, ring_radius + ring_thickness + 10, 3):
            x = int(center[0] + r * np.cos(rad))
            y = int(center[1] + r * np.sin(rad))
            cv.circle(img, (x, y), 2, 220 + np.random.randint(-20, 20), -1)
    
    # Add some Gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def test_full_unroll_pipeline():
    """Test the complete unroll pipeline with a synthetic image."""
    print("Creating synthetic embryo image...")
    I = create_synthetic_embryo(size=600, ring_radius=200, ring_thickness=60)
    
    print(f"Input image shape: {I.shape}")
    print(f"Input image dtype: {I.dtype}")
    
    print("\nProcessing with CVImage...")
    try:
        cv_image = CVImage(
            I=I,
            id=1.0,
            size=(512, 512),
            padding=44,
            plot_images=False,
            npoints=100,
            inward=40,
            outward=-24
        )
        
        print("✓ CVImage initialization successful")
        
        # Check outputs
        print(f"\nPreprocessed image shape: {cv_image.I.shape}")
        print(f"Segmentation mask shape: {cv_image.Ilabel.shape}")
        print(f"Number of border points: {len(cv_image.x)}")
        print(f"Unrolled image shape: {cv_image.Inl.shape}")
        print(f"Final image shape: {cv_image.image.shape}")
        
        # Validate dimensions
        expected_depth = abs(cv_image.inward) + abs(cv_image.outward)
        if cv_image.Inl.shape[0] == expected_depth:
            print(f"\n✓ Unrolled depth correct: {cv_image.Inl.shape[0]} == {expected_depth}")
        else:
            print(f"\n✗ Unrolled depth incorrect: {cv_image.Inl.shape[0]} != {expected_depth}")
            return False
        
        # Check that dimensions match between preprocessing and segmentation
        if cv_image.I.shape == cv_image.Ilabel.shape:
            print(f"✓ Segmentation dimensions match: {cv_image.I.shape}")
        else:
            print(f"✗ Dimension mismatch: I={cv_image.I.shape}, Ilabel={cv_image.Ilabel.shape}")
            return False
        
        # Check that border points are within image bounds
        h, w = cv_image.I.shape
        if (np.all(cv_image.x >= 0) and np.all(cv_image.x < w) and
            np.all(cv_image.y >= 0) and np.all(cv_image.y < h)):
            print(f"✓ Border points within bounds")
        else:
            print(f"✗ Some border points out of bounds")
            return False
        
        print("\n" + "="*50)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_unroll_pipeline()
    sys.exit(0 if success else 1)
