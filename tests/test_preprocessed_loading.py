#!/usr/bin/env python3
"""
Test script for the GPU-accelerated preprocessed image loading pipeline.

This script loads images using the TorchDataset with use_preprocessed=True
and saves sample images to disk for manual inspection.

Usage:
    python test_preprocessed_loading.py
"""

import os
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.torchdataset import TorchDataset


def load_config(config_path: str = 'config.yml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_tensor_as_image(tensor: torch.Tensor, filepath: str, title: str = ""):
    """
    Save a PyTorch tensor as an image file.
    
    Args:
        tensor: Image tensor [C, H, W] in range [0, 1]
        filepath: Output file path
        title: Optional title for the image
    """
    # Convert tensor to numpy and squeeze channel dimension if grayscale
    img_np = tensor.cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)  # Remove channel dimension for grayscale
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.colorbar(label='Intensity [0-1]')
    
    # Save
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")


def main():
    """Main test function."""
    print("=" * 80)
    print("Testing Preprocessed Image Loading Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config('config.yml')
    
    # Extract relevant settings
    data_config = config['data']
    model_config = config['model']
    system_config = config['system']
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Data path: {data_config['paths']['root']}")
    print(f"use_preprocessed: {data_config['loading']['use_preprocessed']}")
    print(f"img_height: {data_config['img_height']}")
    print(f"img_width: {data_config['img_width']}")
    print(f"trunc_width: {data_config['loading']['trunc_width']}")
    
    # Create output directory
    output_dir = Path('./test_output_preprocessed')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Create dataset
    print("\nCreating TorchDataset...")
    dataset = TorchDataset(
        path=data_config['paths']['root'],
        test=data_config['splits']['test_ids'],
        val=data_config['splits']['val_ids'],
        ignore=data_config['splits'].get('ignore_ids', []),
        size=(data_config['img_height'], data_config['img_width']),
        trunc_width=data_config['loading']['trunc_width'],
        metadata_path=data_config['paths']['metadata'],
        data_augment=data_config['augmentation']['enabled'],
        use_preprocessed=data_config['loading']['use_preprocessed'],
        type='train',
        device=device  # Load directly to GPU
    )
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Load and save 10 sample images
    num_samples = min(10, len(dataset))
    print(f"\nLoading {num_samples} sample images...")
    
    for i in range(num_samples):
        try:
            # Load image through the pipeline
            image_tensor, target, folder_id = dataset[i]
            
            # Get image information
            folder, idx = dataset.indices[i]
            height, width = image_tensor.shape[1], image_tensor.shape[2]
            
            # Create descriptive filename and title
            filename = f"sample_{i:02d}_folder{folder_id}_idx{idx}.png"
            title = (f"Sample {i} | Folder: {folder} (ID: {folder_id}) | "
                    f"Target: {target:.4f} | Shape: [{height}, {width}]")
            
            # Save image
            filepath = output_dir / filename
            save_tensor_as_image(image_tensor, str(filepath), title)
            
            # Print info
            print(f"  [{i+1}/{num_samples}] Folder: {folder}, Target: {target:.4f}, "
                  f"Shape: {image_tensor.shape}, Device: {image_tensor.device}")
            
        except Exception as e:
            print(f"  ERROR loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"\nReview the saved images in: {output_dir.absolute()}")
    print("\nWhat to check:")
    print("  1. Images should be properly loaded and visible")
    print("  2. Image dimensions should match config settings")
    print("  3. Intensity range should be [0, 1]")
    print("  4. No artifacts or corruption")
    print("  5. Random crops should be applied (if trunc_width < img_width)")
    print("\nIf images look correct, the GPU-accelerated pipeline is working! ✓")


if __name__ == "__main__":
    main()
