#!/usr/bin/env python3
"""
Prepare training data by augmenting training images and copying test/val images.

This script generates synthetic training images by interpolating between adjacent
images using the same augmentation strategy as get_random_image_from_folder_idx.
Test and validation images are copied as-is without augmentation.

Usage:
    # Using config file (default: scripts/training_data.yml)
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --unroll

    # Using custom config file
    python scripts/prepare_training_data.py --config /path/to/config.yml

    # Using command line arguments (overrides config file)
    python scripts/prepare_training_data.py \
        --data_path /path/to/source/data \
        --output_path /path/to/output/data \
        --num_augmentations 10 \
        --test_ids 21 22 \
        --val_ids 6 34 \
        --unroll
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import cv2 as cv
import torch
from tqdm import tqdm
import yaml

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cvimage import CVImage
from src.data import Data


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
    parser.add_argument(
        "--unroll",
        action='store_true',
        help="Unroll images using CVImage before saving (requires config with unroll parameters)"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to metadata CSV file (overrides config)"
    )
    parser.add_argument(
        "--augment_distribution",
        type=str,
        default=None,
        choices=['uniform', 'beta'],
        help="Distribution for augmentation sampling (overrides config, default: uniform)"
    )
    parser.add_argument(
        "--augment_beta_alpha",
        type=float,
        default=None,
        help="Alpha parameter for Beta distribution (overrides config)"
    )
    parser.add_argument(
        "--augment_beta_beta",
        type=float,
        default=None,
        help="Beta parameter for Beta distribution (overrides config)"
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help="Enable parallel processing (overrides config)"
    )
    parser.add_argument(
        "--no_parallel",
        action='store_true',
        help="Disable parallel processing (overrides config)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (overrides config, default: auto-detect)"
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


def unroll_image(
    image: np.ndarray,
    image_id: float,
    folder_name: str,
    unroll_config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Unroll an image using CVImage.

    Args:
        image: Input grayscale image
        image_id: Image identifier/stage value
        folder_name: Name of the folder (for determining sagittal vs cross-section)
        unroll_config: Configuration dictionary with CVImage parameters
        metadata: Optional metadata dictionary with PPM info

    Returns:
        Unrolled image as numpy array
    """
    folder_id = get_folder_id(folder_name)

    # Get boundary parameters based on folder type
    sagittal_prefixes = unroll_config.get('sagittal_folder_prefixes', [6, 7])
    if folder_id in sagittal_prefixes:
        boundary = unroll_config['boundary_extension']['sagittal']
    else:
        boundary = unroll_config['boundary_extension']['cross_section']

    # Get PPM from metadata if available
    source_ppm = metadata.get('ppm', None) if metadata else None

    # Get target_ppm from config (None disables PPM scaling)
    target_ppm = unroll_config.get('target_ppm', None)

    # Create CVImage instance
    cv_image = CVImage(
        I=image,
        id=image_id,
        size=unroll_config['size'],
        padding=unroll_config['padding'],
        plot_images=False,
        npoints=unroll_config['npoints'],
        inward=boundary['inward'],
        outward=boundary['outward'],
        trunc_width=None,  # Don't truncate when saving to disk
        source_ppm=source_ppm,
        target_ppm=target_ppm
    )

    # Get unrolled image
    unrolled = cv_image.get_unrolled_image(trunc_width=None)

    # Squeeze to remove channel dimension if present
    if unrolled.ndim == 3 and unrolled.shape[2] == 1:
        unrolled = unrolled.squeeze(axis=2)

    # Enforce final image height if specified
    final_height = unroll_config.get('final_image_height', None)
    if final_height is not None:
        current_height = unrolled.shape[0]
        
        if current_height > final_height:
            raise ValueError(
                f"Unrolled image height ({current_height}) exceeds final_image_height ({final_height}). "
                f"Folder: {folder_name}, Image ID: {image_id}"
            )
        elif current_height < final_height:
            # Extend top border with top pixel row
            padding_needed = final_height - current_height
            unrolled = cv.copyMakeBorder(
                unrolled,
                top=padding_needed,
                bottom=0,
                left=0,
                right=0,
                borderType=cv.BORDER_CONSTANT,
                value=0
            )

    return unrolled


def copy_folder(
    data: Data,
    folder: str,
    dst_folder: Path,
    list_type: str,
    unroll: bool = False,
    unroll_config: Optional[Dict[str, Any]] = None,
    folder_metadata: Optional[Dict[str, Any]] = None
):
    """
    Copy a folder and all its contents, including id.csv.

    Args:
        data: Data instance for loading images
        folder: Folder name (e.g., "6_name")
        dst_folder: Destination folder path
        list_type: Dataset type ('train', 'test', or 'val')
        unroll: Whether to unroll images before saving
        unroll_config: Configuration for unrolling (required if unroll=True)
        folder_metadata: Metadata for the folder (for PPM scaling)
    """
    print(f"  Copying {folder}...")

    # Create destination folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Get the data dictionary for this list type
    list_dict = {'train': data.train_data, 'test': data.test_data, 'val': data.val_data}
    df = list_dict[list_type][folder]

    # Get source folder path
    src_folder = Path(data.data_path) / folder

    if not unroll:
        # Simple copy without unrolling
        for file in src_folder.iterdir():
            if file.is_file():
                shutil.copy2(file, dst_folder / file.name)
    else:
        # Copy and unroll images using Data class
        if unroll_config is None:
            raise ValueError("unroll_config required when unroll=True")

        # Process each image using Data class
        for idx in tqdm(range(len(df)), desc=f"    Unrolling {folder}"):
            # Load image using Data class
            I, img_id = data.get_raw_image(folder, idx, list_type)

            # Unroll image
            I_unrolled = unroll_image(I, img_id, folder, unroll_config, folder_metadata)

            # Get original filename to save with same name
            img_path = df.iloc[idx, 0]  # Full path from id.csv
            img_filename = Path(img_path).name

            # Save unrolled image
            dst_img_path = dst_folder / img_filename
            cv.imwrite(str(dst_img_path), I_unrolled)

        # Copy id.csv
        shutil.copy2(src_folder / 'id.csv', dst_folder / 'id.csv')


def generate_augmented_images(
    data: Data,
    folder: str,
    dst_folder: Path,
    num_augmentations: int,
    unroll: bool = False,
    unroll_config: Optional[Dict[str, Any]] = None,
    folder_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate augmented images for a training folder using Data class pipeline.

    For each adjacent pair of images (i, i+1) that forms a valid interpolation pair
    (matching s, c, z with consecutive t), generates num_augmentations synthetic images
    using uniform grid sampling (evenly spaced alpha values between 0 and 1).

    Args:
        data: Data instance for loading and augmenting images
        folder: Folder name (e.g., "1_name")
        dst_folder: Destination folder path
        num_augmentations: Number of synthetic images per image pair
        unroll: Whether to unroll images before saving
        unroll_config: Configuration for unrolling (required if unroll=True)
        folder_metadata: Metadata for the folder (for PPM scaling)
    """
    if unroll:
        print(f"  Augmenting and unrolling {folder}...")
    else:
        print(f"  Augmenting {folder}...")

    if unroll and unroll_config is None:
        raise ValueError("unroll_config required when unroll=True")

    # Create destination folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Get folder name for path prefixes
    folder_name = dst_folder.name

    # Get dataframe for this folder
    df = data.train_data[folder]

    # Prepare new DataFrame for augmented data
    new_rows = []

    # Process each adjacent pair of images
    for idx in tqdm(range(len(df) - 1), desc=f"    Processing {folder}"):
        # Load first original image using Data class
        I1, id1 = data.get_raw_image(folder, idx, 'train')

        # Get original filename for naming
        img1_path = df.iloc[idx, 0]  # Full path from id.csv
        base_name1 = Path(img1_path).stem
        ext1 = Path(img1_path).suffix

        # Process and save first original image
        dst_img1_name = f"{base_name1}_orig{ext1}"
        dst_img1_path = dst_folder / dst_img1_name

        if unroll:
            I1_processed = unroll_image(I1, id1, folder, unroll_config, folder_metadata)
        else:
            I1_processed = I1

        cv.imwrite(str(dst_img1_path), I1_processed)
        new_rows.append({'path': f"{folder_name}/{dst_img1_name}", 'id': id1})

        # Generate augmented images using uniform grid sampling
        augmented_images = data.get_augmented_images_from_folder_idx(
            folder, idx, num_augmentations, 'train'
        )
        
        # Save each augmented image
        for aug_idx, (I_aug, id_aug) in enumerate(augmented_images):
            # Unroll if requested
            if unroll:
                I_aug_processed = unroll_image(I_aug, id_aug, folder, unroll_config, folder_metadata)
            else:
                I_aug_processed = I_aug

            # Save augmented image
            aug_name = f"{base_name1}_aug_{aug_idx}.png"
            aug_path = dst_folder / aug_name
            cv.imwrite(str(aug_path), I_aug_processed)

            # Add to new rows (include folder name in path to match original format)
            new_rows.append({'path': f"{folder_name}/{aug_name}", 'id': id_aug})

    # Add the last original image
    if len(df) > 0:
        # Load last image using Data class
        I_last, last_id = data.get_raw_image(folder, len(df) - 1, 'train')

        # Get original filename
        last_img_path = df.iloc[-1, 0]
        base_name_last = Path(last_img_path).stem
        ext_last = Path(last_img_path).suffix
        dst_img_last_name = f"{base_name_last}_orig{ext_last}"
        dst_img_last_path = dst_folder / dst_img_last_name

        if unroll:
            I_last_processed = unroll_image(I_last, last_id, folder, unroll_config, folder_metadata)
        else:
            I_last_processed = I_last

        cv.imwrite(str(dst_img_last_path), I_last_processed)
        new_rows.append({'path': f"{folder_name}/{dst_img_last_name}", 'id': last_id})

    # Save new id.csv
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(dst_folder / 'id.csv', index=False, header=False)

    print(f"    Generated {len(new_rows)} total images ({len(df)} original + {len(new_rows) - len(df)} augmented)")


def load_metadata(metadata_path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """
    Load metadata from CSV file.

    Args:
        metadata_path: Path to metadata CSV file

    Returns:
        Dictionary mapping folder IDs to metadata dictionaries
    """
    if metadata_path is None or not metadata_path.exists():
        return {}

    df = pd.read_csv(metadata_path)
    metadata = {}

    for _, row in df.iterrows():
        # Extract folder_id from Filename column (e.g., "1_..." -> 1)
        filename = row['Filename']
        folder_id = int(filename.split('_')[0])

        metadata[folder_id] = {
            'ppm': row.get('ppm', None)
        }

    return metadata


def process_single_folder(
    folder: str,
    data_loader: Data,
    output_path: Path,
    test_ids_set: set,
    val_ids_set: set,
    num_augmentations: int,
    unroll: bool,
    unroll_config: Optional[Dict[str, Any]],
    metadata_dict: Dict[int, Dict[str, Any]]
) -> Tuple[str, bool]:
    """
    Process a single folder (for parallel execution).

    Args:
        folder: Folder name to process
        data_loader: Data instance for loading images
        output_path: Base output path
        test_ids_set: Set of test folder IDs
        val_ids_set: Set of validation folder IDs
        num_augmentations: Number of augmentations per pair
        unroll: Whether to unroll images
        unroll_config: Configuration for unrolling
        metadata_dict: Metadata dictionary for folders

    Returns:
        Tuple of (folder_name, success_status)
    """
    try:
        folder_id = get_folder_id(folder)
        dst_folder = output_path / folder
        folder_metadata = metadata_dict.get(folder_id, {})

        if folder_id in test_ids_set:
            print(f"Test folder: {folder}")
            copy_folder(data_loader, folder, dst_folder, 'test', unroll=unroll,
                       unroll_config=unroll_config, folder_metadata=folder_metadata)
        elif folder_id in val_ids_set:
            print(f"Validation folder: {folder}")
            copy_folder(data_loader, folder, dst_folder, 'val', unroll=unroll,
                       unroll_config=unroll_config, folder_metadata=folder_metadata)
        else:
            print(f"Training folder: {folder}")
            generate_augmented_images(data_loader, folder, dst_folder, num_augmentations,
                                     unroll=unroll, unroll_config=unroll_config,
                                     folder_metadata=folder_metadata)
        return (folder, True)
    except Exception as e:
        print(f"Error processing {folder}: {e}")
        import traceback
        traceback.print_exc()
        return (folder, False)


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
    unroll = args.unroll
    metadata_path = args.metadata_path if args.metadata_path is not None else config.get('metadata_path')
    augment_distribution = args.augment_distribution if args.augment_distribution is not None else config.get('augment_distribution', 'uniform')
    augment_beta_alpha = args.augment_beta_alpha if args.augment_beta_alpha is not None else config.get('augment_beta_alpha', 0.5)
    augment_beta_beta = args.augment_beta_beta if args.augment_beta_beta is not None else config.get('augment_beta_beta', 0.5)

    # Parse parallel processing config
    parallel_config = config.get('parallel_processing', {})
    if args.parallel:
        use_parallel = True
    elif args.no_parallel:
        use_parallel = False
    else:
        use_parallel = parallel_config.get('enable', False)
    
    num_workers = args.num_workers if args.num_workers is not None else parallel_config.get('num_workers', None)
    if num_workers is None:
        num_workers = mp.cpu_count()

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

    # Build unroll config if unrolling is enabled
    unroll_config = None
    metadata_dict = {}
    if unroll:
        # Try to load unroll parameters from config
        unroll_params = config.get('unroll_params', {})

        # Required parameters with defaults
        unroll_config = {
            'size': tuple(unroll_params.get('size', [512, 512])),
            'padding': unroll_params.get('padding', 44),
            'npoints': unroll_params.get('npoints', 100),
            'boundary_extension': unroll_params.get('boundary_extension', {
                'cross_section': {'inward': 34, 'outward': -30},
                'sagittal': {'inward': 34, 'outward': -30}
            }),
            'sagittal_folder_prefixes': unroll_params.get('sagittal_folder_prefixes', [6, 7]),
            'target_ppm': unroll_params.get('target_ppm', None),  # None disables PPM scaling
            'final_image_height': unroll_params.get('final_image_height', None)  # None disables height enforcement
        }

        # Load metadata if available
        if metadata_path is not None:
            metadata_dict = load_metadata(Path(metadata_path))
            print(f"Loaded metadata for {len(metadata_dict)} folders")

    # Initialize Data class for loading images
    print("\nInitializing Data loader...")
    data_loader = Data(
        path=str(data_path),
        test=list(test_ids) if test_ids else [],
        val=list(val_ids) if val_ids else [],
        metadata_path=metadata_path,
        augment_distribution=augment_distribution,
        augment_beta_alpha=augment_beta_alpha,
        augment_beta_beta=augment_beta_beta
    )

    # Get test and val IDs
    test_ids_set = set(test_ids) if test_ids else set()
    val_ids_set = set(val_ids) if val_ids else set()

    print(f"\nPreparing training data...")
    print(f"Source: {data_path}")
    print(f"Output: {output_path}")
    print(f"Augmentations per pair: {num_augmentations}")
    print(f"Augmentation distribution: {augment_distribution}")
    if augment_distribution == 'beta':
        print(f"Beta parameters: alpha={augment_beta_alpha}, beta={augment_beta_beta}")
    print(f"Test IDs: {sorted(test_ids_set) if test_ids_set else 'None'}")
    print(f"Val IDs: {sorted(val_ids_set) if val_ids_set else 'None'}")
    print(f"Seed: {seed}")
    print(f"Parallel processing: {use_parallel}")
    if use_parallel:
        print(f"Number of workers: {num_workers}")
    print(f"Unroll: {unroll}")
    if unroll:
        target_ppm = unroll_config.get('target_ppm', None)
        if target_ppm is not None:
            print(f"PPM scaling: Enabled (target_ppm={target_ppm})")
        else:
            print(f"PPM scaling: Disabled")
        if metadata_path:
            print(f"Metadata: {metadata_path}")
    print()

    # Process each folder using Data class
    all_folders = data_loader.train_list + data_loader.test_list + data_loader.val_list
    all_folders_sorted = sorted(all_folders)

    if use_parallel and len(all_folders_sorted) > 1:
        # Parallel processing using multiprocessing
        print(f"Processing {len(all_folders_sorted)} folders in parallel with {num_workers} workers...\n")
        
        # Create partial function with fixed arguments
        process_func = partial(
            process_single_folder,
            data_loader=data_loader,
            output_path=output_path,
            test_ids_set=test_ids_set,
            val_ids_set=val_ids_set,
            num_augmentations=num_augmentations,
            unroll=unroll,
            unroll_config=unroll_config,
            metadata_dict=metadata_dict
        )
        
        # Process folders in parallel
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_func, all_folders_sorted)
        
        # Check for failures
        failed_folders = [folder for folder, success in results if not success]
        if failed_folders:
            print(f"\n⚠ Warning: {len(failed_folders)} folder(s) failed to process: {failed_folders}")
    else:
        # Sequential processing
        if use_parallel:
            print("Only one folder to process, using sequential mode...\n")
        
        for folder in all_folders_sorted:
            folder_id = get_folder_id(folder)
            dst_folder = output_path / folder

            # Get metadata for this folder
            folder_metadata = metadata_dict.get(folder_id, {})

            if folder_id in test_ids_set:
                print(f"Test folder: {folder}")
                copy_folder(data_loader, folder, dst_folder, 'test', unroll=unroll,
                           unroll_config=unroll_config, folder_metadata=folder_metadata)
            elif folder_id in val_ids_set:
                print(f"Validation folder: {folder}")
                copy_folder(data_loader, folder, dst_folder, 'val', unroll=unroll,
                           unroll_config=unroll_config, folder_metadata=folder_metadata)
            else:
                print(f"Training folder: {folder}")
                generate_augmented_images(data_loader, folder, dst_folder, num_augmentations,
                                         unroll=unroll, unroll_config=unroll_config,
                                         folder_metadata=folder_metadata)

    print(f"\n✓ Data preparation complete!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
