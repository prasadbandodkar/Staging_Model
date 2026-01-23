"""
Image Processing Pipeline Script

Comprehensive script for the image processing pipeline.
Processes real embryo images from data folders with two modes:
1. 'all': Process images and save all intermediate processing stages (debugging/validation)
2. 'final': Process images and save only the final image specified in process_config.yml

Configuration:
    Uses process_config.yml to specify:
    - Data paths (root directory, metadata file)
    - Image type to save (original, segmented, nuclear_layer, or unrolled)
    - Preprocessing parameters (dimensions, padding, boundary extension)

Usage:
    # Run with default config settings
    python scripts/process_images.py
    
    # Run with custom config file
    python scripts/process_images.py --config my_config.yml
    
    # Override specific config settings via command line
    python scripts/process_images.py --folder "34_" --mode all
    python scripts/process_images.py --output-dir ./custom_output --limit 5
"""

import os
import sys
import json
import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import cv2 as cv
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cvimage import CVImage

# Simple config class for process_images.py
class ProcessConfig:
    """Simplified configuration for image processing script."""
    def __init__(self, config_dict: dict):
        # Data paths
        self.data_root = config_dict['data']['root']
        self.data_metadata = config_dict['data']['metadata']
        
        # Execution settings
        exec_cfg = config_dict.get('execution', {})
        self.folder = exec_cfg.get('folder', None)
        self.all = exec_cfg.get('all', True)
        self.output_dir = exec_cfg.get('output_dir', './script_output')
        self.mode = exec_cfg.get('mode', 'final')
        self.limit = exec_cfg.get('limit', None)
        self.workers = exec_cfg.get('workers', cpu_count())
        self.no_parallel = exec_cfg.get('no_parallel', False)
        
        # Preprocessing parameters
        self.img_height = config_dict['preprocessing']['img_height']
        self.img_width = config_dict['preprocessing']['img_width']
        self.padding = config_dict['preprocessing']['padding']
        self.image_type = config_dict['preprocessing']['image_type']
        self.npoints = config_dict['preprocessing']['npoints']
        self.target_ppm = config_dict['preprocessing']['target_ppm']
        self.sagittal_folder_prefixes = config_dict['preprocessing']['sagittal_folder_prefixes']
        
        # Parse boundary extension
        be = config_dict['preprocessing']['boundary_extension']
        self.cross_section_inward = be['cross_section']['inward']
        self.cross_section_outward = be['cross_section']['outward']
        self.sagittal_inward = be['sagittal']['inward']
        self.sagittal_outward = be['sagittal']['outward']
    
    def get_boundary_params(self, folder_id: int) -> tuple:
        """Get boundary parameters based on folder ID."""
        if folder_id in self.sagittal_folder_prefixes:
            return self.sagittal_inward, self.sagittal_outward
        else:
            return self.cross_section_inward, self.cross_section_outward
    
    @classmethod
    def load(cls, config_path: str = "Scripts/process_config.yml") -> "ProcessConfig":
        """Load configuration from YAML file."""
        import yaml
        from pathlib import Path
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)


def get_valid_folders(data_path: Path) -> List[str]:
    """
    Get list of valid data folders by scanning the data directory.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        List of folder names
    """
    folders = []
    if not data_path.exists():
        return folders
    
    for item in sorted(data_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if folder contains image files
            image_files = get_image_files(item)
            if image_files:
                folders.append(item.name)
    
    return folders


def get_image_files(folder_path: Path) -> List[Path]:
    """
    Get list of image files in a folder.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = []
    
    for file_path in sorted(folder_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


def load_image_and_id(image_path: Path) -> Tuple[np.ndarray, float]:
    """
    Load an image and extract its ID from the filename.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (image: np.ndarray, id: float)
    """
    # Load image as grayscale (CVImage expects single-channel images)
    I = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if I is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Use hash of full filename to ensure uniqueness
    # This prevents images with different z-slices but same time point from overwriting
    # (e.g., s1_c2_z1_t42.png vs s1_c2_z2_t42.png)
    stem = image_path.stem
    id = float(abs(hash(stem)) % 100000)
    
    return I, id


def process_image(
    image_path: Path,
    folder_name: str,
    output_dir: Path,
    config: ProcessConfig,
    mode: str,
    metadata: Dict[int, Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Process a single image through the pipeline based on the specified mode.
    
    Args:
        image_path: Path to the image file
        folder_name: Name of the folder containing the image
        output_dir: Directory to save results
        config: Processing configuration
        mode: Processing mode ('all' or 'final')
               - 'all': Save all intermediate processing steps
               - 'final': Save only the final image specified by config.image_type
        metadata: Optional metadata dict mapping folder_id to {ppm, pixel_type}
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # Load image and extract ID
        I, id = load_image_and_id(image_path)
        
        # Create folder-specific output directory
        folder_output_dir = output_dir / folder_name
        folder_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get folder-specific boundary parameters
        # Extract folder ID from folder name (e.g., "6_emb" -> 6)
        folder_id = int(folder_name.split('_')[0])
        inward, outward = config.get_boundary_params(folder_id)
        
        # Get metadata for this folder (PPM scaling)
        source_ppm = None
        if metadata and folder_id in metadata:
            source_ppm = metadata[folder_id].get('ppm', None)
        
        # In 'all' mode, we pass output_dir to CVImage for automatic saving of intermediate steps.
        # In 'final' mode, we do NOT pass output_dir, so CVImage doesn't save anything automatically.
        cv_image_output_dir = folder_output_dir if mode == 'all' else None
        
        # Create CVImage instance with folder-specific parameters
        cv_image = CVImage(
            I=I,
            id=id,
            size=(config.img_height, config.img_width),
            padding=config.padding,
            plot_images=False,
            npoints=config.npoints,
            inward=inward,
            outward=outward,
            output_dir=cv_image_output_dir,
            filename_stem=image_path.stem,
            source_ppm=source_ppm,
            target_ppm=config.target_ppm
        )
        
        # Get the image based on the configured image type
        img = cv_image.get_image(image_type=config.image_type)
        
        if img is None:
            return False, "Image processing failed - no output"
        
        # In 'final' mode, we need to explicitly save the final result
        # (In 'all' mode, intermediate images are already saved by CVImage)
        if mode == 'final' or (mode == 'all' and config.image_type != 'unrolled'):
            # Create appropriate suffix based on image type
            suffix_map = {
                'original': 'original',
                'segmented': 'segmented', 
                'nuclear_layer': 'nuclear_layer',
                'unrolled': 'unrolled'
            }
            suffix = suffix_map.get(config.image_type, config.image_type)
            output_filename = f"{image_path.stem}_{suffix}.png"
            output_path = folder_output_dir / output_filename
            
            # Squeeze channel dimension if needed
            img_to_save = img.squeeze() if len(img.shape) == 3 and img.shape[2] == 1 else img
            cv.imwrite(str(output_path), img_to_save)
            
        return True, None
        
    except Exception as e:
        return False, str(e)


def process_folder(
    data_path: Path,
    folder_name: str,
    output_dir: Path,
    config: ProcessConfig,
    mode: str,
    metadata: Dict[int, Dict] = None,
    limit: Optional[int] = None
) -> Dict[str, any]:
    """
    Process all images in a folder.
    
    Args:
        data_path: Path to the data directory
        folder_name: Name of folder to process
        output_dir: Output directory for results
        config: Processing configuration
        mode: Processing mode ('all' or 'final')
        metadata: Optional metadata dict mapping folder_id to {ppm, pixel_type}
        limit: Optional limit on number of images to process
        
    Returns:
        Dictionary with processing statistics including timing
    """
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*70}")
    
    stats = {
        'folder': folder_name,
        'total': 0,
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    # Get image files from this folder
    folder_path = data_path / folder_name
    if not folder_path.exists():
        print(f"  ✗ Folder not found: {folder_path}")
        return stats
    
    image_files = get_image_files(folder_path)
    if not image_files:
        print(f"  ✗ No images found in folder")
        return stats
    
    # Limit number of images if requested
    num_images = min(len(image_files), limit) if limit else len(image_files)
    stats['total'] = num_images
    
    print(f"  Processing {num_images} images...")
    
    # Process each image
    for idx, img_file in enumerate(image_files[:num_images]):
        success, error = process_image(
            image_path=img_file,
            folder_name=folder_name,
            output_dir=output_dir,
            config=config,
            mode=mode,
            metadata=metadata
        )
        
        # Extract ID for display purposes
        try:
            _, id = load_image_and_id(img_file)
        except:
            id = idx
        
        if success:
            stats['success'] += 1
            print(f"    ✓ {idx + 1}/{num_images} - {img_file.name} (ID {id:06.2f})")
        else:
            stats['failed'] += 1
            stats['errors'].append({
                'filename': img_file.name,
                'index': idx,
                'error': error
            })
            print(f"    ✗ {idx + 1}/{num_images} - {img_file.name} - Error: {error}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    stats['elapsed_time_seconds'] = elapsed_time
    
    # Print folder summary
    success_rate = 100 * stats['success'] / stats['total'] if stats['total'] > 0 else 0
    print(f"\n  Summary:")
    print(f"    Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    if stats['failed'] > 0:
        print(f"    Failed:  {stats['failed']}")
    print(f"    Time:    {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    
    return stats


def process_folder_wrapper(args: Tuple[Path, str, Path, ProcessConfig, str, Dict[int, Dict], Optional[int]]) -> Dict[str, any]:
    """
    Wrapper function for multiprocessing pool.
    
    Args:
        args: Tuple of (data_path, folder_name, output_dir, config, mode, metadata, limit)
        
    Returns:
        Processing statistics dictionary
    """
    data_path, folder_name, output_dir, config, mode, metadata, limit = args
    return process_folder(data_path, folder_name, output_dir, config, mode, metadata, limit)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Image processing pipeline - all settings can be configured in process_config.yml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run with default config
        python scripts/process_images.py
        
        # Run with custom config file
        python scripts/process_images.py --config my_config.yml
        
        # Override config settings via command line
        python scripts/process_images.py --folder "34_" --mode all
        python scripts/process_images.py --output-dir ./custom_output --limit 5
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/process_config.yml',
        help='Path to configuration file (default: scripts/process_config.yml)'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Process a single folder by name (overrides config)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        default=None,
        help='Process all valid data folders (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for images (overrides config)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'final'],
        default=None,
        help='Processing mode (overrides config)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of images per folder (overrides config)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (overrides config)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        default=None,
        help='Disable parallel processing (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = ProcessConfig.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Override config with command line arguments (if provided)
    if args.folder is not None:
        config.folder = args.folder
        config.all = False
    if args.all is not None and args.all:
        config.all = True
        config.folder = None
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.mode is not None:
        config.mode = args.mode
    if args.limit is not None:
        config.limit = args.limit
    if args.workers is not None:
        config.workers = args.workers
    if args.no_parallel is not None:
        config.no_parallel = args.no_parallel
    
    # Validate settings
    if not config.folder and not config.all:
        parser.error("Must specify either --folder or --all (in config or command line)")
    
    if config.folder and config.all:
        parser.error("Cannot specify both --folder and --all")
    
    # Create output directory (delete if exists for clean slate)
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data path from config
    data_path = Path(config.data_root)
    print(f"Loading data from: {data_path}")
    print(f"Image type to save: {config.image_type}")
    
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    # Load metadata if available
    metadata = {}
    if config.data_metadata:
        try:
            import csv
            with open(config.data_metadata, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['Filename']
                    if not filename:
                        continue
                    folder_id_str = filename.split('_')[0]
                    try:
                        folder_id = int(folder_id_str)
                    except ValueError:
                        continue
                    ppm_str = row.get('ppm', '').strip()
                    ppm = float(ppm_str) if ppm_str else None
                    metadata[folder_id] = {
                        'filename': filename,
                        'pixel_type': row.get('Pixel Type', 'uint16'),
                        'ppm': ppm
                    }
            print(f"Loaded metadata for {len(metadata)} folders")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            metadata = {}
    
    # Get folders to process
    if config.folder:
        # Support partial folder name matching
        valid_folders = get_valid_folders(data_path)
        
        # First try exact match
        if config.folder in valid_folders:
            folders_to_process = [config.folder]
        else:
            # Try prefix match (case-insensitive) - folder must START with the pattern
            matching_folders = [f for f in valid_folders if f.lower().startswith(config.folder.lower())]
            
            if not matching_folders:
                print(f"Error: No folder starting with '{config.folder}' found")
                print(f"Valid folders: {', '.join(valid_folders)}")
                sys.exit(1)
            elif len(matching_folders) == 1:
                folders_to_process = matching_folders
                print(f"Matched folder: {matching_folders[0]}")
            else:
                print(f"Error: Multiple folders match '{config.folder}':")
                for f in matching_folders:
                    print(f"  - {f}")
                print(f"\nPlease be more specific.")
                sys.exit(1)
    else:
        folders_to_process = get_valid_folders(data_path)
        print(f"Found {len(folders_to_process)} valid folders")
    
    # Process folders
    print(f"\nStarting processing in '{config.mode}' mode with {config.workers if not config.no_parallel else 1} worker(s)...")
    overall_start_time = time.time()
    
    all_stats = []
    
    if config.no_parallel or len(folders_to_process) == 1:
        # Sequential processing
        for folder_name in folders_to_process:
            stats = process_folder(
                data_path=data_path,
                folder_name=folder_name,
                output_dir=output_dir,
                config=config,
                mode=config.mode,
                metadata=metadata,
                limit=config.limit
            )
            all_stats.append(stats)
    else:
        # Parallel processing with multiprocessing
        pool_args = [
            (data_path, folder_name, output_dir, config, config.mode, metadata, config.limit)
            for folder_name in folders_to_process
        ]
        
        with Pool(processes=config.workers) as pool:
            all_stats = pool.map(process_folder_wrapper, pool_args)
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_images = sum(s['total'] for s in all_stats)
    total_success = sum(s['success'] for s in all_stats)
    total_failed = sum(s['failed'] for s in all_stats)
    overall_success_rate = 100 * total_success / total_images if total_images > 0 else 0
    
    print(f"Mode:              {config.mode}")
    print(f"Folders processed: {len(all_stats)}")
    print(f"Total images:      {total_images}")
    print(f"Successful:        {total_success} ({overall_success_rate:.1f}%)")
    print(f"Failed:            {total_failed}")
    print(f"\nTiming:")
    print(f"  Overall time:    {overall_elapsed_time:.2f}s ({overall_elapsed_time/60:.2f} min)")
    if total_images > 0:
        print(f"  Time per image:  {overall_elapsed_time/total_images:.2f}s")
    print(f"  Workers used:    {config.workers if not config.no_parallel else 1}")
    print(f"\nOutput saved to: {output_dir.absolute()}")
    
    # Save summary as JSON
    summary_path = output_dir / 'summary.json'
    summary_data = {
        'config': {
            'data_path': str(config.data_root),
            'image_type': config.image_type,
            'img_size': (config.img_height, config.img_width),
            'padding': config.padding,
            'npoints': config.npoints,
            'boundary_extension': {
                'cross_section': {
                    'inward': config.cross_section_inward,
                    'outward': config.cross_section_outward
                },
                'sagittal': {
                    'inward': config.sagittal_inward,
                    'outward': config.sagittal_outward
                }
            },
            'sagittal_folder_prefixes': config.sagittal_folder_prefixes,
            'mode': config.mode
        },
        'timing': {
            'overall_time_seconds': overall_elapsed_time,
            'overall_time_minutes': overall_elapsed_time / 60,
            'time_per_image_seconds': overall_elapsed_time / total_images if total_images > 0 else 0,
            'workers_used': config.workers if not config.no_parallel else 1
        },
        'overall': {
            'folders': len(all_stats),
            'total_images': total_images,
            'success': total_success,
            'failed': total_failed,
            'success_rate': overall_success_rate
        },
        'folders': all_stats
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}\n")
    
    # Exit with appropriate code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
