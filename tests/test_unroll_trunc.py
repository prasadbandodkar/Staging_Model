"""
Test script for extracting truncated and augmented nuclear layer images.

This script:
1. Loads a single test image from the dataset
2. Extracts the full unrolled nuclear layer using CVImage
3. Generates 10 truncated versions at different positions
4. Applies augmentations using TorchImage to each
5. Saves all 10 augmented images to output/test_unroll_trunc/
"""

import numpy as np
import cv2 as cv
import torch
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.cvimage import CVImage
from src.torchimage import TorchImage
from src.config import AppConfig


def main(input_folder=None, output_folder=None, image_id=None):
    """Main test function.
    
    Args:
        input_folder: Optional path to folder containing images (overrides config)
        output_folder: Optional output directory path (overrides default)
        image_id: Optional image ID/index to select specific image (0-based index)
    """
    print("=" * 80)
    print("TEST: Truncated and Augmented Nuclear Layer Extraction")
    print("=" * 80)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yml'
    config = AppConfig.load(str(config_path))
    
    print(f"\n1. Configuration loaded from: {config_path}")
    print(f"   - Image size: {config.data.img_height}x{config.data.img_width}")
    print(f"   - Padding: {config.data.padding}")
    print(f"   - Truncation width: {config.data.trunc_width}")
    print(f"   - Number of points: {config.data.npoints}")
    
    # Set random seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Determine data path (use custom folder if provided, otherwise use config)
    if input_folder:
        data_path = Path(input_folder)
        if not data_path.exists():
            print(f"\n❌ ERROR: Input folder does not exist: {data_path}")
            return
        print(f"   - Using custom input folder: {data_path}")
        
        # If input_folder is a directory containing images directly
        if list(data_path.glob('*.png')) or list(data_path.glob('*.tif*')):
            # It's a folder with images directly
            test_folder = data_path
        else:
            # It's a parent folder, get first subfolder
            folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
            if not folders:
                print(f"\n❌ ERROR: No subfolders found in {data_path}")
                return
            test_folder = folders[0]
    else:
        data_path = Path(config.data.path)
        folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
        
        if not folders:
            print(f"\n❌ ERROR: No folders found in {data_path}")
            return
        
        test_folder = folders[0]
    
    # Find images in the folder
    images = sorted(test_folder.glob('*.png'))
    if not images:
        images = sorted(test_folder.glob('*.tif*'))
    
    if not images:
        print(f"\n❌ ERROR: No images found in {test_folder}")
        return
    
    # Select image by ID or use first one
    if image_id is not None:
        if 0 <= image_id < len(images):
            test_image_path = images[image_id]
            print(f"   - Selected image ID {image_id} out of {len(images)} images")
        else:
            print(f"\n❌ ERROR: Image ID {image_id} out of range (0-{len(images)-1})")
            return
    else:
        test_image_path = images[0]
    print(f"\n2. Selected test image: {test_image_path.name}")
    print(f"   From folder: {test_folder.name}")
    
    # Extract folder ID for boundary parameters
    folder_name = test_folder.name
    try:
        folder_id = int(folder_name.split('_')[0])
        boundary_params = config.data.get_boundary_params(folder_id)
    except (ValueError, IndexError):
        # If folder name doesn't start with ID, use default cross-section params
        print(f"   ⚠️  Could not extract folder ID from '{folder_name}', using default cross-section params")
        folder_id = 0
        boundary_params = config.data.cross_section
    
    print(f"   - Folder ID: {folder_id}")
    print(f"   - Boundary extension: inward={boundary_params.inward}, outward={boundary_params.outward}")
    
    # Load image
    I = cv.imread(str(test_image_path), cv.IMREAD_GRAYSCALE)
    if I is None:
        print(f"\n❌ ERROR: Failed to load image: {test_image_path}")
        return
    
    print(f"   - Image shape: {I.shape}")
    print(f"   - Image dtype: {I.dtype}")
    
    # Process image with CVImage
    print(f"\n3. Processing image with CVImage...")
    cv_image = CVImage(
        I=I,
        id=1.0,
        size=(config.data.img_height, config.data.img_width),
        padding=config.data.padding,
        plot_images=False,
        npoints=config.data.npoints,
        inward=boundary_params.inward,
        outward=boundary_params.outward
    )
    
    # Get truncation width from config
    trunc_width = config.data.trunc_width
    num_samples = 10
    
    print(f"\n4. Generating {num_samples} truncated versions using pipeline truncation...")
    print(f"   - Truncation width: {trunc_width}")
    print(f"   - Each call to get_unrolled_image() will randomly truncate")
    
    # Create output directory
    if output_folder:
        output_dir = Path(output_folder)
    else:
        output_dir = Path(__file__).parent.parent / 'output' / 'test_unroll_trunc'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n5. Output directory: {output_dir}")
    
    # Save the full unrolled nuclear layer (no truncation)
    print(f"\n6. Saving original full nuclear layer...")
    full_nuclear_layer = cv_image.get_image(image_type='unrolled', trunc_width=None)
    
    # Process through TorchImage (automatically normalizes to [0,1])
    torch_image_full = TorchImage(np.array(full_nuclear_layer, dtype=np.float32), id=0)
    # No augmentation or additional normalization needed
    I_full_tensor = torch_image_full.I.squeeze().cpu().numpy()
    I_full_to_save = (I_full_tensor * 255).astype(np.uint8)
    
    # Save full nuclear layer
    full_output_path = output_dir / "nuclear_layer_full.png"
    cv.imwrite(str(full_output_path), I_full_to_save)
    print(f"   Saved: nuclear_layer_full.png (shape={I_full_to_save.shape})")
    
    # Process each truncated version
    print(f"\n7. Extracting, processing, and saving truncated images...")
    print(f"   (Following exact training pipeline: get_image → TorchImage → normalize → save)")
    
    for idx in range(num_samples):
        # EXACT TRAINING PIPELINE (CORRECTED):
        # Step 1: Get processed image using get_image() with truncation
        processed_image = cv_image.get_image(image_type='unrolled', trunc_width=trunc_width)
        
        # Step 2: Create TorchImage - _prepare_tensor normalizes to [0,1] automatically
        torch_image = TorchImage(np.array(processed_image, dtype=np.float32), id=idx)
        
        # Step 3: Apply augmentation (operates on [0,1] range)
        # Enable this for training mode (uncomment to test augmentation):
        torch_image.I = torch_image.augment(seed=config.seed + idx)
        
        # Step 4: Image is already in [0,1] range - no additional normalization needed
        # (The old normalize() call was redundant)
        
        # Step 5: Convert back to numpy for saving
        I_tensor = torch_image.I.squeeze().cpu().numpy()
        
        # Step 6: Convert from [0,1] float to [0,255] uint8 for saving
        I_to_save = (I_tensor * 255).astype(np.uint8)
        
        # Save image
        output_filename = f"truncated_{idx:02d}.png"
        output_path = output_dir / output_filename
        cv.imwrite(str(output_path), I_to_save)
        
        print(f"   [{idx+1:2d}/{num_samples}] Saved: {output_filename} (shape={I_to_save.shape})")
    
    print("\n" + "=" * 80)
    print(f"✓ SUCCESS: {num_samples} augmented truncated images saved to:")
    print(f"  {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract truncated and augmented nuclear layer images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Use defaults (first image from config data path)
            python tests/test_unroll_trunc.py
            
            # Specify custom input folder
            python tests/test_unroll_trunc.py --input-folder /path/to/images
            
            # Specify custom output folder
            python tests/test_unroll_trunc.py --output-folder /path/to/output
            
            # Select specific image by ID (0-based index)
            python tests/test_unroll_trunc.py --image-id 5
            
            # Combine all options
            python tests/test_unroll_trunc.py --input-folder /path/to/images --output-folder /path/to/output --image-id 3
                    """
    )
    
    parser.add_argument(
        '--input-folder', '-i',
        type=str,
        default=None,
        help='Path to folder containing images (overrides config.yml data path). Can be a parent folder or direct image folder.'
    )
    
    parser.add_argument(
        '--output-folder', '-o',
        type=str,
        default=None,
        help='Output directory for saving truncated images (default: output/test_unroll_trunc/)'
    )
    
    parser.add_argument(
        '--image-id', '-n',
        type=int,
        default=None,
        help='Image ID/index to process (0-based). If not specified, uses first image.'
    )
    
    args = parser.parse_args()
    
    try:
        main(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            image_id=args.image_id
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
