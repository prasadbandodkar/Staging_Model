#!/usr/bin/env python3
"""
Convert all TIFF images in a folder (including subfolders) to PNG format.
Also updates id.csv files to replace .tiff extensions with .png.

Usage:
    python convert_tiff_to_png.py <input_folder> [--output_folder <output_folder>] [--keep_original]

Arguments:
    input_folder: Path to the folder containing TIFF images
    --output_folder: Optional. Path to save PNG images. If not specified, saves in same location as original
    --keep_original: Optional. Keep original TIFF files (default is to delete them)
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Set, Dict


def find_tiff_files(root_dir: str) -> List[Path]:
    """
    Find all TIFF files in the given directory and its subdirectories.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of Path objects for all TIFF files found
    """
    tiff_extensions = {'.tiff', '.tif', '.TIFF', '.TIF'}
    tiff_files = []
    
    root_path = Path(root_dir)
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in tiff_extensions:
            tiff_files.append(file_path)
    
    return tiff_files


def find_id_csv_files(root_dir: str) -> List[Path]:
    """
    Find all id.csv files in the given directory and its subdirectories.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of Path objects for all id.csv files found
    """
    id_csv_files = []
    root_path = Path(root_dir)
    for file_path in root_path.rglob('id.csv'):
        if file_path.is_file():
            id_csv_files.append(file_path)
    
    return id_csv_files


def update_id_csv(csv_path: Path) -> Tuple[bool, str, int]:
    """
    Update an id.csv file to replace .tiff/.tif extensions with .png in the first column.
    
    Args:
        csv_path: Path to the id.csv file
        
    Returns:
        Tuple of (success: bool, message: str, replacements: int)
    """
    try:
        # Read the CSV file
        rows = []
        replacements = 0
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) > 0:
                    # Replace .tiff or .tif with .png in the first column
                    original = row[0]
                    # Case-insensitive replacement
                    for ext in ['.tiff', '.tif', '.TIFF', '.TIF']:
                        if original.endswith(ext):
                            row[0] = original[:-len(ext)] + '.png'
                            replacements += 1
                            break
                rows.append(row)
        
        # Write back to the CSV file
        if replacements > 0:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return True, f"Updated {csv_path} ({replacements} replacements)", replacements
        else:
            return True, f"No changes needed in {csv_path}", 0
            
    except Exception as e:
        return False, f"Error updating {csv_path}: {str(e)}", 0


def convert_tiff_to_png(
    tiff_path: Path, 
    output_folder: Path = None,
    delete_original: bool = True
) -> Tuple[bool, str]:
    """
    Convert a single TIFF file to PNG, overwriting if it already exists.
    
    Args:
        tiff_path: Path to the TIFF file
        output_folder: Optional output folder. If None, saves in same location as original
        delete_original: Whether to delete the original TIFF file after conversion
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Open the TIFF image
        with Image.open(tiff_path) as img:
            # Determine output path
            if output_folder:
                # Maintain the same relative directory structure
                relative_path = tiff_path.relative_to(tiff_path.parents[len(tiff_path.parents) - 1])
                png_path = output_folder / relative_path.with_suffix('.png')
                
                # Create output directory if it doesn't exist
                png_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Save in the same location as the original
                png_path = tiff_path.with_suffix('.png')
            
            # Check if PNG already exists
            already_exists = png_path.exists()
            
            # Convert and save as PNG (overwrite if exists)
            # Handle different image modes appropriately
            if img.mode in ('RGBA', 'LA', 'PA'):
                # Already has alpha channel
                img.save(png_path, 'PNG')
            elif img.mode == 'P':
                # Palette mode - convert to RGBA to preserve transparency if present
                img = img.convert('RGBA')
                img.save(png_path, 'PNG')
            else:
                # RGB, L, or other modes
                img.save(png_path, 'PNG')
            
            # Delete original if requested
            if delete_original:
                tiff_path.unlink()
                action = "Converted (overwrote), deleted" if already_exists else "Converted, deleted"
                return True, f"{action}: {tiff_path} -> {png_path}"
            else:
                action = "Converted (overwrote)" if already_exists else "Converted"
                return True, f"{action}: {tiff_path} -> {png_path}"
                
    except Exception as e:
        return False, f"Error converting {tiff_path}: {str(e)}"


def main():
    """Main function to handle command-line arguments and perform conversion."""
    parser = argparse.ArgumentParser(
        description='Convert all TIFF images in a folder (including subfolders) to PNG format. '
                    'Also updates id.csv files to use .png extensions.'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to the folder containing TIFF images'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Optional. Path to save PNG images. If not specified, saves in same location as original'
    )
    parser.add_argument(
        '--keep_original',
        action='store_true',
        help='Keep original TIFF files (default is to delete them after conversion)'
    )
    
    args = parser.parse_args()
    
    # By default, delete originals (unless --keep_original is specified)
    delete_original = not args.keep_original
    
    # Validate input folder
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: '{args.input_folder}' is not a directory.")
        sys.exit(1)
    
    # Prepare output folder if specified
    output_path = Path(args.output_folder) if args.output_folder else None
    if output_path and not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all TIFF files
    print(f"Searching for TIFF files in '{input_path}'...")
    tiff_files = find_tiff_files(input_path)
    
    if not tiff_files:
        print("No TIFF files found.")
    else:
        print(f"Found {len(tiff_files)} TIFF file(s). Starting conversion...")
        
        # Convert each file
        success_count = 0
        fail_count = 0
        
        for i, tiff_file in enumerate(tiff_files, 1):
            success, message = convert_tiff_to_png(
                tiff_file, 
                output_path, 
                delete_original
            )
            
            if success:
                success_count += 1
                print(f"[{i}/{len(tiff_files)}] ✓ {message}")
            else:
                fail_count += 1
                print(f"[{i}/{len(tiff_files)}] ✗ {message}")
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"Image conversion complete!")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Total: {len(tiff_files)}")
        print("=" * 60 + "\n")
    
    # Find and update id.csv files
    print(f"Searching for id.csv files in '{input_path}'...")
    id_csv_files = find_id_csv_files(input_path)
    
    if not id_csv_files:
        print("No id.csv files found.")
    else:
        print(f"Found {len(id_csv_files)} id.csv file(s). Updating...")
        
        csv_success_count = 0
        csv_fail_count = 0
        total_replacements = 0
        
        for i, csv_file in enumerate(id_csv_files, 1):
            success, message, replacements = update_id_csv(csv_file)
            
            if success:
                csv_success_count += 1
                total_replacements += replacements
                print(f"[{i}/{len(id_csv_files)}] ✓ {message}")
            else:
                csv_fail_count += 1
                print(f"[{i}/{len(id_csv_files)}] ✗ {message}")
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"CSV update complete!")
        print(f"Files updated: {csv_success_count}")
        print(f"Total replacements: {total_replacements}")
        print(f"Failed: {csv_fail_count}")
        print("=" * 60)


if __name__ == '__main__':
    main()
