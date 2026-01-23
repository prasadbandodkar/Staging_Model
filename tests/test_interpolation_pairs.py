#!/usr/bin/env python3
"""
Test script to verify that interpolation only happens between
images with matching s, c, z values and consecutive t values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import Data

def test_interpolation_pairs():
    """Test that interpolation pairs are correctly validated."""
    
    # Initialize Data object - using test directory with uniform image sizes
    data_path = "/Volumes/X2/Projects/phd/staging/data/live_data/"
    
    print("Initializing Data object...")
    data = Data(
        path=data_path,
        test=[6, 7],
        val=[21, 34],
        ignore=[]
    )
    
    print(f"\nTrain folders: {data.train_list[:3]}...")
    print(f"Total train folders: {len(data.train_list)}")
    
    # Test on the specific folder
    test_folder = "1_1_Embryo_1_end_of_nc12-gastrulation_czi"
    if test_folder in data.train_data:
        print(f"\n{'='*60}")
        print(f"Testing folder: {test_folder}")
        print(f"{'='*60}")
        
        # Build valid pairs
        valid_pairs = data._build_valid_interpolation_pairs(test_folder, 'train')
        total_images = len(data.train_data[test_folder])
        
        print(f"\nTotal images in folder: {total_images}")
        print(f"Valid interpolation pairs found: {len(valid_pairs)}")
        print(f"Percentage of valid pairs: {100*len(valid_pairs)/(total_images-1):.1f}%")
        
        # Show some examples of valid pairs
        if valid_pairs:
            print(f"\nFirst 5 valid pair starting indices: {valid_pairs[:5]}")
            
            # Verify a few pairs
            print(f"\n{'='*60}")
            print("Verifying first 3 pairs:")
            print(f"{'='*60}")
            
            for i, idx in enumerate(valid_pairs[:3]):
                filename1 = data.train_data[test_folder].iloc[idx, 0]
                filename2 = data.train_data[test_folder].iloc[idx + 1, 0]
                
                meta1 = data._parse_filename_metadata(filename1)
                meta2 = data._parse_filename_metadata(filename2)
                
                print(f"\nPair {i+1} (idx={idx}):")
                print(f"  File 1: {os.path.basename(filename1)}")
                print(f"          s={meta1['s']}, c={meta1['c']}, z={meta1['z']}, t={meta1['t']}")
                print(f"  File 2: {os.path.basename(filename2)}")
                print(f"          s={meta2['s']}, c={meta2['c']}, z={meta2['z']}, t={meta2['t']}")
                
                # Verify
                assert meta1['s'] == meta2['s'], "Stage (s) values don't match!"
                assert meta1['c'] == meta2['c'], "Channel (c) values don't match!"
                assert meta1['z'] == meta2['z'], "Z-slice (z) values don't match!"
                assert meta2['t'] == meta1['t'] + 1, "Timepoints (t) are not consecutive!"
                print(f"  ✓ Valid: s, c, z match and t values are consecutive")
        
        # Test sampling random images
        print(f"\n{'='*60}")
        print("Testing random image sampling:")
        print(f"{'='*60}")
        
        for i in range(3):
            I, id, idx = data.get_random_image_from_folder(test_folder, 'train')
            filename1 = data.train_data[test_folder].iloc[idx, 0]
            filename2 = data.train_data[test_folder].iloc[idx + 1, 0]
            
            print(f"\nRandom sample {i+1}:")
            print(f"  Starting idx: {idx}")
            print(f"  File 1: {os.path.basename(filename1)}")
            print(f"  File 2: {os.path.basename(filename2)}")
            print(f"  Interpolated ID: {id:.2f}")
            print(f"  Image shape: {I.shape}")
        
        print(f"\n{'='*60}")
        print("✓ All tests passed!")
        print(f"{'='*60}")
    else:
        print("No training folders found!")

if __name__ == "__main__":
    test_interpolation_pairs()
