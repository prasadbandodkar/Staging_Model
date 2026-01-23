#!/usr/bin/env python3
"""
Test script to verify the new get_augmented_images_from_folder_idx function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import Data

def test_grid_augmentation():
    """Test that grid-based augmentation creates evenly-spaced IDs."""
    
    # Initialize Data object
    data_path = "/Volumes/X2/Projects/phd/staging/data/live_data/"
    
    print("Initializing Data object...")
    data = Data(
        path=data_path,
        test=[6, 7],
        val=[21, 34],
        ignore=[]
    )
    
    # Test on specific folder
    test_folder = "1_1_Embryo_1_end_of_nc12-gastrulation_czi"
    
    if test_folder not in data.train_data:
        print(f"❌ Folder {test_folder} not found")
        return
    
    print(f"\nTesting folder: {test_folder}")
    
    # Find a valid interpolation pair
    valid_pairs = data._build_valid_interpolation_pairs(test_folder, 'train')
    
    if not valid_pairs:
        print("❌ No valid pairs found")
        return
    
    print(f"Found {len(valid_pairs)} valid pairs")
    
    # Test with the first valid pair
    idx = valid_pairs[0]
    num_augmentations = 5
    
    print(f"\n{'='*60}")
    print(f"Testing grid augmentation:")
    print(f"  Pair index: {idx}")
    print(f"  Num augmentations: {num_augmentations}")
    print(f"{'='*60}")
    
    # Get original images
    I1, id1 = data.get_raw_image(test_folder, idx, 'train')
    I2, id2 = data.get_raw_image(test_folder, idx + 1, 'train')
    
    print(f"\nOriginal images:")
    print(f"  Image 1 ID: {id1:.6f}")
    print(f"  Image 2 ID: {id2:.6f}")
    print(f"  ID range: {id2 - id1:.6f}")
    
    # Generate augmented images
    augmented = data.get_augmented_images_from_folder_idx(
        test_folder, idx, num_augmentations, 'train'
    )
    
    print(f"\nAugmented images (uniform grid):")
    for i, (img, img_id) in enumerate(augmented):
        # Calculate expected alpha for uniform grid
        # For n=5: alpha = [1/6, 2/6, 3/6, 4/6, 5/6]
        expected_alpha = (i + 1) / (num_augmentations + 1)
        expected_id = expected_alpha * id1 + (1 - expected_alpha) * id2
        
        print(f"  Aug {i}: ID = {img_id:.6f} (expected: {expected_id:.6f}, alpha: {expected_alpha:.4f})")
        print(f"         Shape: {img.shape}")
        
        # Verify ID is within expected range
        assert id1 <= img_id <= id2 or id2 <= img_id <= id1, f"ID {img_id} out of range [{id1}, {id2}]"
        
        # Verify ID matches expected value (within floating point precision)
        assert abs(img_id - expected_id) < 1e-10, f"ID mismatch: {img_id} != {expected_id}"
    
    print(f"\n✓ All augmented IDs are evenly spaced!")
    print(f"✓ Grid augmentation working correctly!")
    
    # Test invalid pair (should return empty list)
    print(f"\n{'='*60}")
    print("Testing invalid pair handling:")
    print(f"{'='*60}")
    
    # Use an idx that's not a valid pair
    invalid_idx = 0
    if invalid_idx not in valid_pairs:
        augmented_invalid = data.get_augmented_images_from_folder_idx(
            test_folder, invalid_idx, num_augmentations, 'train'
        )
        print(f"  Invalid pair at idx={invalid_idx}: returned {len(augmented_invalid)} images")
        assert len(augmented_invalid) == 0, "Should return empty list for invalid pair"
        print(f"  ✓ Correctly returns empty list for invalid pairs")
    
    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_grid_augmentation()
