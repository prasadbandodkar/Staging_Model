#!/usr/bin/env python3
"""
Prepare training data by augmenting training images and copying test/val images.

This script generates synthetic training images by interpolating between adjacent
images using the same augmentation strategy as get_random_image_from_folder_idx.
Test and validation images are copied as-is without augmentation.

Usage:
    # Using config file (default: scripts/training_data.yml)
    python scripts/prepare_training_data.py

    # Using custom config file
    python scripts/prepare_training_data.py --config /path/to/config.yml

    # Using command line arguments (overrides config file)
    python scripts/prepare_training_data.py \
        --data_path /path/to/source/data \
        --output_path /path/to/output/data \
        --num_augmentations 10 \
        --test_ids 21 22 \
        --val_ids 6 34
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare training data with augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: scripts/training_data.yml)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to source data directory (overrides config)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output data directory (overrides config)"
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=None,
        help="Number of synthetic images to generate per image pair (overrides config)"
    )
    parser.add_argument(
        "--test_ids",
        type=int,
        nargs='*',
        default=None,
        help="List of folder IDs for testing (overrides config, e.g., --test_ids 21 22)"
    )
    parser.add_argument(
        "--val_ids",
        type=int,
        nargs='*',
        default=None,
        help="List of folder IDs for validation (overrides config, e.g., --val_ids 6 34)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)"
    )
    return parser.parse_args()


def get_folder_id(folder_name: str) -> int:
    """
    Extract numeric ID from folder name (e.g., '6_name' -> 6).

    Args:
        folder_name: Folder name string

    Returns:
        Numeric folder ID
    """
    return int(folder_name.split('_')[0])


def load_id_csv(folder_path: Path) -> pd.DataFrame:
    """
    Load the id.csv file from a folder.

    Args:
        folder_path: Path to the folder

    Returns:
        DataFrame with image paths and IDs
    """
    id_csv_path = folder_path / 'id.csv'
    if not id_csv_path.exists():
        raise FileNotFoundError(f"id.csv not found in {folder_path}")

    df = pd.read_csv(id_csv_path, header=None, names=['path', 'id'])
    return df


def copy_folder(src_folder: Path, dst_folder: Path):
    """
    Copy a folder and all its contents, including id.csv.

    Args:
        src_folder: Source folder path
        dst_folder: Destination folder path
    """
    print(f"  Copying {src_folder.name}...")

    # Create destination folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Copy all files
    for file in src_folder.iterdir():
        if file.is_file():
            shutil.copy2(file, dst_folder / file.name)


def generate_augmented_images(
    src_folder: Path,
    dst_folder: Path,
    num_augmentations: int
) -> None:
    """
    Generate augmented images for a training folder.

    For each adjacent pair of images (i, i+1), generates num_augmentations
    synthetic images using random interpolation with Beta(0.5, 0.5) distribution.

    Args:
        src_folder: Source folder path
        dst_folder: Destination folder path
        num_augmentations: Number of synthetic images per image pair
    """
    print(f"  Augmenting {src_folder.name}...")

    # Create destination folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Get folder name for path prefixes
    folder_name = dst_folder.name

    # Load id.csv
    df = load_id_csv(src_folder)

    # Prepare new DataFrame for augmented data
    new_rows = []

    # Process each adjacent pair of images
    for idx in tqdm(range(len(df) - 1), desc=f"    Processing {src_folder.name}"):
        # Get image pair info
        img1_rel_path = df.iloc[idx]['path']
        img2_rel_path = df.iloc[idx + 1]['path']
        id1 = df.iloc[idx]['id']
        id2 = df.iloc[idx + 1]['id']

        # Load images (extract just the filename from the relative path)
        img1_path = src_folder / Path(img1_rel_path).name
        img2_path = src_folder / Path(img2_rel_path).name

        if not img1_path.exists():
            print(f"    Warning: {img1_path} not found, skipping...")
            continue
        if not img2_path.exists():
            print(f"    Warning: {img2_path} not found, skipping...")
            continue

        I1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE)
        I2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE)

        if I1 is None or I2 is None:
            print(f"    Warning: Could not load images for pair {idx}, skipping...")
            continue

        # Copy original images first
        base_name1 = Path(img1_rel_path).stem
        ext1 = Path(img1_rel_path).suffix
        dst_img1_name = f"{base_name1}_orig{ext1}"
        dst_img1_path = dst_folder / dst_img1_name
        cv.imwrite(str(dst_img1_path), I1)
        # Include folder name in path to match original format
        new_rows.append({'path': f"{folder_name}/{dst_img1_name}", 'id': id1})

        # Generate augmented images
        for aug_idx in range(num_augmentations):
            # Sample alpha from Beta(0.5, 0.5) - same as in get_random_image_from_folder_idx
            alpha = torch.distributions.Beta(0.5, 0.5).sample().item()

            # Interpolate images
            I_aug = cv.addWeighted(I1, alpha, I2, 1 - alpha, 0)

            # Calculate interpolated ID
            id_aug = alpha * id1 + (1 - alpha) * id2

            # Save augmented image
            aug_name = f"{base_name1}_aug_{aug_idx}.png"
            aug_path = dst_folder / aug_name
            cv.imwrite(str(aug_path), I_aug)

            # Add to new rows (include folder name in path to match original format)
            new_rows.append({'path': f"{folder_name}/{aug_name}", 'id': id_aug})

    # Add the last original image
    if len(df) > 0:
        last_img_rel_path = df.iloc[-1]['path']
        last_id = df.iloc[-1]['id']
        last_img_path = src_folder / Path(last_img_rel_path).name

        if last_img_path.exists():
            I_last = cv.imread(str(last_img_path), cv.IMREAD_GRAYSCALE)
            if I_last is not None:
                base_name_last = Path(last_img_rel_path).stem
                ext_last = Path(last_img_rel_path).suffix
                dst_img_last_name = f"{base_name_last}_orig{ext_last}"
                dst_img_last_path = dst_folder / dst_img_last_name
                cv.imwrite(str(dst_img_last_path), I_last)
                # Include folder name in path to match original format
                new_rows.append({'path': f"{folder_name}/{dst_img_last_name}", 'id': last_id})

    # Save new id.csv
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(dst_folder / 'id.csv', index=False, header=False)

    print(f"    Generated {len(new_rows)} total images ({len(df)} original + {len(new_rows) - len(df)} augmented)")


def main():
    """Main execution function."""
    args = parse_args()

    # Determine config file path
    if args.config is not None:
        config_path = Path(args.config)
    else:
        # Default to scripts/training_data.yml relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir / 'training_data.yml'

    # Load config if it exists
    config = {}
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        print(f"Warning: Config file not found at {config_path}")
        print("Using command-line arguments only")

    # Merge config with command-line arguments (CLI takes precedence)
    data_path = args.data_path if args.data_path is not None else config.get('data_path')
    output_path = args.output_path if args.output_path is not None else config.get('output_path')
    num_augmentations = args.num_augmentations if args.num_augmentations is not None else config.get('num_augmentations')
    test_ids = args.test_ids if args.test_ids is not None else config.get('test_ids')
    val_ids = args.val_ids if args.val_ids is not None else config.get('val_ids')
    seed = args.seed if args.seed is not None else config.get('seed', 42)

    # Validate required parameters
    if data_path is None:
        raise ValueError("data_path must be specified in config file or command line")
    if output_path is None:
        raise ValueError("output_path must be specified in config file or command line")
    if num_augmentations is None:
        raise ValueError("num_augmentations must be specified in config file or command line")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert paths to Path objects
    data_path = Path(data_path)
    output_path = Path(output_path)

    # Validate input path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get test and val IDs
    test_ids_set = set(test_ids) if test_ids else set()
    val_ids_set = set(val_ids) if val_ids else set()

    print(f"\nPreparing training data...")
    print(f"Source: {data_path}")
    print(f"Output: {output_path}")
    print(f"Augmentations per pair: {num_augmentations}")
    print(f"Test IDs: {sorted(test_ids_set) if test_ids_set else 'None'}")
    print(f"Val IDs: {sorted(val_ids_set) if val_ids_set else 'None'}")
    print(f"Seed: {seed}")
    print()

    # Get all folders
    folders = [f for f in data_path.iterdir() if f.is_dir()]

    if not folders:
        raise ValueError(f"No folders found in {data_path}")

    # Process each folder
    for folder in sorted(folders):
        folder_id = get_folder_id(folder.name)
        src_folder = folder
        dst_folder = output_path / folder.name

        if folder_id in test_ids_set:
            print(f"Test folder: {folder.name}")
            copy_folder(src_folder, dst_folder)
        elif folder_id in val_ids_set:
            print(f"Validation folder: {folder.name}")
            copy_folder(src_folder, dst_folder)
        else:
            print(f"Training folder: {folder.name}")
            generate_augmented_images(src_folder, dst_folder, num_augmentations)

    print(f"\n✓ Data preparation complete!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
