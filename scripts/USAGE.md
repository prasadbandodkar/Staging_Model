# Image Processing Pipeline - Usage Guide

This guide describes how to use the `process_images.py` script to process embryo images through the complete pipeline.

## Overview

The script processes embryo images from data folders with two modes:

- **all**: Process and save all intermediate processing stages (debugging/validation)
- **unroll**: Process and save only the final unrolled nuclear layer (production)

## Running with uv

This project uses `uv` for dependency management. You can run the script directly without activating a virtual environment:

```bash
uv run python scripts/process_images.py [OPTIONS]
```

All the examples below can be prefixed with `uv run` to run them in the managed environment. For instance:

```bash
# Instead of:
python scripts/process_images.py --folder "6_emb" --output-dir ./output/full --mode all

# Use:
uv run python scripts/process_images.py --folder "6_emb" --output-dir ./output/full --mode all
```

This ensures that all dependencies are properly installed and the correct Python environment is used.

## Command-Line Options

### Required Options (choose one)

| Option | Description |
|--------|-------------|
| `--folder FOLDER` | Process a single folder by name (e.g., `"6_emb"`). Supports partial name matching - the folder name must START with the pattern (case-insensitive). |
| `--all` | Process all valid data folders found in the configured data directory. |

### Optional Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir DIR` | `./script_output` | Output directory where processed images will be saved. **Warning:** Existing directory will be deleted and recreated. |
| `--mode {all,unroll}` | `unroll` | Processing mode:<br>• `all` - Save all intermediate steps (preprocessed, segmented, border, extended borders, unrolled)<br>• `unroll` - Save only the final unrolled output |
| `--limit N` | None | Limit the number of images to process per folder. Useful for quick tests. |
| `--workers N` | CPU count | Number of parallel workers for folder processing. By default, uses all available CPU cores. |
| `--no-parallel` | false | Disable parallel processing and process folders sequentially (useful for debugging). |

## Usage Examples

### Basic Usage

Process a specific folder with all intermediate outputs:

```bash
python scripts/process_images.py --folder "6_emb" --output-dir ./output/full --mode all
```

Process a specific folder with only final unrolled output:

```bash
python scripts/process_images.py --folder "6_emb" --output-dir ./output/unrolled --mode unroll
```

### Partial Folder Name Matching

Process a folder using partial name (must match from the start):

```bash
python scripts/process_images.py --folder "6_" --output-dir ./output/folder6
```

If multiple folders start with "6_", the script will list them and ask you to be more specific.

### Process All Folders

Process all folders, save only unrolled images:

```bash
python scripts/process_images.py --all --output-dir ./output/complete --mode unroll
```

Process all folders with full debugging output:

```bash
python scripts/process_images.py --all --output-dir ./output/debug --mode all
```

### Quick Testing

Process only the first 3 images from each folder:

```bash
python scripts/process_images.py --all --limit 3 --output-dir ./output/quick_test
```

Process a single folder with 5 images:

```bash
python scripts/process_images.py --folder "6_emb" --limit 5 --output-dir ./output/sample
```

### Performance Tuning

Use 4 parallel workers:

```bash
python scripts/process_images.py --all --workers 4 --output-dir ./output/parallel4
```

Process folders sequentially (no parallelization):

```bash
python scripts/process_images.py --all --no-parallel --output-dir ./output/sequential
```

## Output Structure

The script creates the following directory structure:

```
output-dir/
├── folder_name_1/
│   ├── image1_stem_01_preprocessed.png    (mode: all only)
│   ├── image1_stem_02_segmentation.png    (mode: all only)
│   ├── image1_stem_03_border.png          (mode: all only)
│   ├── image1_stem_04_borders_extended.png (mode: all only)
│   ├── image1_stem_05_unrolled.png        (mode: all only)
│   └── image1_stem_unrolled.png           (mode: unroll)
├── folder_name_2/
│   └── ...
└── summary.json
```

### Output Files

#### Mode: `all`

Saves five intermediate processing stages for each image:

1. **01_preprocessed.png** - Resized and padded image
2. **02_segmentation.png** - Complete segmentation pipeline visualization (8-step process)
3. **03_border.png** - Detected and smoothed embryo border
4. **04_borders_extended.png** - Extended inner and outer nuclear layer boundaries
5. **05_unrolled.png** - Final unrolled nuclear layer

#### Mode: `unroll`

Saves only:

- **{stem}_unrolled.png** - Final unrolled nuclear layer

#### Summary File

The `summary.json` file contains:

- Configuration parameters used
- Timing statistics (overall time, per-image time, workers used)
- Processing results for each folder (success/failure counts)
- Detailed error messages for any failed images

## Configuration

The script reads parameters from `config.yml`:

- Image dimensions and padding
- Nuclear layer boundary parameters (inward/outward distances)
- Data folder path
- Metadata path (for PPM/pixel scaling)
- Sagittal vs cross-section folder mappings

Folder-specific boundary parameters are automatically selected based on the folder ID (extracted from folder name prefix).

## Metadata Integration

If a metadata CSV file is configured (`data.metadata_path` in `config.yml`), the script will:

1. Load pixel-per-micron (PPM) values for each folder
2. Apply PPM scaling to normalize images to the target resolution (1 pixel per micron by default)
3. Handle cases where PPM data is missing (leaves image unscaled)

## Performance Notes

- **Parallel Processing**: By default, the script uses all CPU cores to process multiple folders in parallel
- **Single Folder**: When processing a single folder, parallelization is automatically disabled
- **Memory**: Each worker processes one folder at a time, so memory usage scales with the number of workers
- **Timing**: The script provides detailed timing statistics including per-image processing time

## Exit Codes

- `0` - All images processed successfully
- `1` - One or more images failed to process (check `summary.json` for details)

## Troubleshooting

**Error: "Must specify either --folder or --all"**

- You must provide one of the two required options

**Error: "Cannot specify both --folder and --all"**

- Choose either a specific folder or all folders, not both

**Error: "No folder starting with 'X' found"**

- The specified folder prefix doesn't match any folders in the data directory
- Use `--all` once to see which folders are available in the summary output

**Error: "Data path does not exist"**

- Check the `data.path` configuration in `config.yml`

**Warning: "Could not load metadata"**

- The script will continue without PPM scaling
- Check that the metadata file path in `config.yml` is correct
