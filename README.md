# Staging Model for Embryo Nuclear Layer Analysis

A comprehensive deep learning pipeline for predicting developmental staging from microscopy images of embryo nuclear layers. This project combines advanced computer vision preprocessing with a ResNet-style convolutional neural network to perform regression on staging values (0-1 range).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Training](#training)
  - [Testing](#testing)
  - [TensorBoard](#tensorboard)
- [Configuration](#configuration)
- [Data Organization](#data-organization)
- [Architecture](#architecture)
  - [Image Processing Pipeline](#image-processing-pipeline)
  - [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Advanced Topics](#advanced-topics)
  - [Run Management](#run-management)
  - [Checkpointing](#checkpointing)
  - [Data Augmentation](#data-augmentation)
  - [Custom Datasets](#custom-datasets)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This project implements an end-to-end machine learning pipeline for developmental staging prediction from embryo microscopy images. The system processes raw microscopy images through a sophisticated computer vision pipeline to extract nuclear layer regions, then uses a deep convolutional neural network to predict continuous staging values.

### Key Capabilities

- **Automated Image Processing**: Handles variable-resolution microscopy images with automatic segmentation, uneven illumination correction, and nuclear layer extraction
- **Deep Learning Regression**: ResNet-style CNN architecture optimized for staging prediction
- **Comprehensive Training Pipeline**: Includes validation, checkpointing, learning rate scheduling, early stopping, and TensorBoard logging
- **Robust MetaData Handling**: Automatically scales images to consistent pixels-per-micron ratios and handles various bit depths
- **Production-Ready**: Built-in run management, experiment tracking, and comprehensive testing tools

---

## Features

### Computer Vision Processing

- **Intelligent Segmentation**: Multi-stage embryo segmentation optimized for speed (256px resolution) with automatic upsampling
- **Illumination Correction**: Automatic detection and correction of uneven illumination using flat-field correction
- **Nuclear Layer Extraction**: Precise border detection with configurable inward/outward extensions
- **Perspective Unrolling**: Transforms curved nuclear layer to flat representation for CNN processing
- **Multi-Resolution Support**: Handles variable-resolution images with automatic PPM (pixels-per-micron) scaling

### Deep Learning

- **ResNet Architecture**: Multiple model sizes (nano, tiny, small, medium, large) for different computational budgets
- **Flexible Loss Functions**: Support for MSE, Smooth L1, and Huber loss
- **Advanced Training**: Gradient clipping, learning rate scheduling, and early stopping
- **Comprehensive Metrics**: Tracks MSE, MAE, RMSE, and R² score during training and validation
- **Data Augmentation**: Random interpolation between adjacent images for synthetic training samples

### Experiment Management

- **Automatic Run Naming**: Timestamp-based or custom run names
- **TensorBoard Integration**: Real-time training visualization, metrics logging, and sample predictions
- **Checkpoint Management**: Automatic best model saving and resumable training
- **Configuration Tracking**: Saves complete configuration with each run
- **Multi-Dataset Testing**: Test on validation, test, both, or custom folder combinations

---

## Installation

### Requirements

- Python ≥ 3.11
- GPU recommended: NVIDIA GPU with CUDA support, or Apple Silicon with MPS
- CPU-only mode also available

### Setup

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd Staging_Model
   ```


2. **Install PyTorch** (platform-specific):

   **For Linux with NVIDIA GPUs (CUDA 11.8)**:

   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   **For Linux with NVIDIA GPUs (CUDA 12.1)**:

   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   **For macOS with Apple Silicon (M1/M2/M3) - MPS acceleration**:

   ```bash
   uv pip install torch torchvision
   ```

   **For macOS Intel or CPU-only systems**:

   ```bash
   uv pip install torch torchvision
   ```

   > **Note**:
   > - Check your CUDA version with `nvidia-smi` on Linux
   > - MPS (Metal Performance Shaders) provides GPU acceleration on Apple Silicon Macs
   > - Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the latest installation commands

3. **Install remaining dependencies**:

   Using `uv`:

   ```bash
   uv pip install -e .
   ```

   Or using standard pip:

   ```bash
   pip install -e .
   ```

4. **Verify installation**:

   Run the verification script to check all dependencies and GPU acceleration:

   ```bash
   python verify_installation.py
   ```

   This will verify:
   - Python version
   - PyTorch installation and version
   - GPU acceleration (CUDA/MPS/CPU)
   - All required dependencies

   You can also verify manually:

   ```bash
   python main.py --help
   ```

   **Check GPU acceleration** (platform-specific):

   For NVIDIA GPUs:

   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

   For Apple Silicon:

   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

### Dependencies

The project requires the following main packages (automatically installed):

- **Deep Learning**: PyTorch ≥2.0.0, torchvision ≥0.15.0
- **Computer Vision**: OpenCV ≥4.8.0, scikit-image ≥0.21.0
- **Scientific Computing**: NumPy ≥1.24.0, SciPy ≥1.11.0
- **Visualization**: Matplotlib ≥3.7.0, TensorBoard ≥2.20.0
- **Data Processing**: Pandas ≥2.0.0, PyYAML ≥6.0.0
- **Utilities**: tqdm ≥4.67.1

---

## Quick Start

### 1. Prepare Your Data

Organize your data according to the [Data Organization](#data-organization) section. Ensure you have:

- Images in folders named with numeric IDs (e.g., `6_experiment1`, `21_control`)
- An `id.csv` file in each folder mapping images to staging values
- A `metadata.csv` file containing bit depth and PPM information

### 2. Configure the Pipeline

Edit `config.yml` to set your data paths and parameters:

```yaml
data:
  path: "/path/to/your/data"
  metadata_path: "/path/to/metadata.csv"
  test_ids: [21, 22]
  val_ids: [6, 34]
```

### 3. Train a Model

```bash
# Train from scratch with automatic run naming
python main.py --mode train

# Train with custom run name
python main.py --mode train --run-name my_first_experiment
```

### 4. Monitor Training

```bash
# Launch TensorBoard for the most recent run
python main.py --mode tensorboard

# Or specify a run name
python main.py --mode tensorboard --run-name my_first_experiment
```

### 5. Test the Model

```bash
# Test on both validation and test sets
python main.py --mode test --checkpoint runs/my_first_experiment/checkpoints/checkpoint_best.pt

# Test only on validation set
python main.py --mode test --checkpoint checkpoint_best.pt --test-on val
```

---

## Detailed Usage

### Training

The training mode creates a complete training pipeline with validation, checkpointing, and logging.

#### Basic Training

```bash
python main.py --mode train
```

This will:

1. Create a new run directory with timestamp (e.g., `runs/2026-01-16_18-30-45`)
2. Save a copy of the configuration
3. Initialize TensorBoard logging
4. Train for the number of epochs specified in `config.yml`
5. Save checkpoints for the best model
6. Log metrics and sample predictions to TensorBoard

#### Custom Run Name

```bash
python main.py --mode train --run-name baseline_experiment
```

#### Resume Training

```bash
python main.py --mode train --resume runs/baseline_experiment/checkpoints/checkpoint_best.pt
```

When resuming:

- Model weights, optimizer state, and epoch counter are restored
- Training continues from the saved epoch
- Metrics history is preserved

#### Training Configuration

Key training parameters in `config.yml`:

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 1.0e-7
  weight_decay: 1.0e-4
  
  # Loss function selection
  loss_type: "mse"  # Options: mse, smooth_l1, huber
  
  # Learning rate scheduler
  lr_factor: 0.5      # Multiply LR by this on plateau
  lr_patience: 5      # Epochs to wait before reducing LR
  
  # Early stopping
  early_stopping: 100  # Stop if no improvement for N epochs
```

### Testing

Testing mode evaluates a trained model on specified datasets and generates comprehensive visualizations.

#### Test on All Data

```bash
python main.py --mode test \
  --checkpoint runs/my_experiment/checkpoints/checkpoint_best.pt
```

#### Test on Specific Dataset

```bash
# Test only
python main.py --mode test --checkpoint checkpoint_best.pt --test-on test

# Validation only
python main.py --mode test --checkpoint checkpoint_best.pt --test-on val
```

#### Test on Custom Folders

```bash
python main.py --mode test \
  --checkpoint checkpoint_best.pt \
  --test-on custom \
  --folders 6 7 21
```

#### Test Outputs

Testing generates:

1. **Console metrics**: MSE, MAE, RMSE, R² score
2. **Scatter plots**: Predicted vs. actual staging values
3. **Saved plots**: In the same directory as the checkpoint

### TensorBoard

Monitor training progress, visualize metrics, and inspect sample predictions in real-time.

#### Launch TensorBoard

```bash
# Latest run
python main.py --mode tensorboard

# Specific run
python main.py --mode tensorboard --run-name my_experiment
```

Access at: `http://localhost:6006`

#### TensorBoard Features

- **Scalars**: Training/validation loss, MAE, RMSE, R², learning rate
- **Images**: Sample predictions with predicted vs. actual values
- **Graphs**: Model architecture visualization
- **Histograms**: Weight and gradient distributions (if enabled)

---

## Configuration

The `config.yml` file controls all aspects of the pipeline. Here's a complete breakdown:

### Data Configuration

```yaml
data:
  # Root directory containing numbered folders with images
  path: "/Volumes/X2/Projects/phd/staging/data/live_data"
  
  # CSV file with bit depth and PPM metadata for each folder
  metadata_path: "/Volumes/X2/Projects/phd/staging/Staging_Model/metadata.csv"
  
  # Folder IDs for test/validation splits
  test_ids: [21, 22]
  val_ids: [6, 34]
  # All other folders automatically become training data
  
  # Target pixels-per-micron for image normalization
  ppm: 5
  
  # Final image dimensions after preprocessing
  img_height: 1024
  img_width: 1024
  
  # Border padding in pixels
  padding: 44
  
  # Number of contour points for border representation
  npoints: 60
  
  # Nuclear layer boundary extension
  # Different settings for cross-section vs sagittal views
  boundary_extension:
    cross_section:
      outward: -20    # Negative → extends outward
      inward: 44      # Positive → extends inward
    sagittal:
      outward: -30
      inward: 34
  
  # Define which folders contain sagittal images
  sagittal_folder_prefixes: [6, 7]
  
  # Width to crop from unrolled image (for data augmentation)
  trunc_width: 128
  
  # Which image type to use for training
  image_type: "unrolled"  # Options: original, segmented, nuclear_layer, unrolled
  
  # Number of workers for data loading (0 = main thread)
  num_workers: 0
```

### Model Configuration

```yaml
model:
  # Model size determines architecture complexity
  model_type: "small"  # Options: nano, tiny, small, medium, large
  
  # Model sizes:
  # - nano:   base_channels=16,  blocks=(1,1,1,1)  ~50K params
  # - tiny:   base_channels=32,  blocks=(1,1,1,1)  ~200K params  
  # - small:  base_channels=64,  blocks=(2,2,2,2)  ~11M params (default)
  # - medium: base_channels=96,  blocks=(2,2,2,2)  ~25M params
  # - large:  base_channels=128, blocks=(3,4,6,3)  ~80M params
  
  dropout: 0.5
  summary: false  # Print model summary at initialization
```

### Training Configuration

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 1.0e-7
  weight_decay: 1.0e-4
  
  # Loss function
  loss_type: "mse"  # Options: mse, smooth_l1, huber
  huber_delta: 1.0  # Only used if loss_type is "huber"
  
  grad_clip: 0.0  # Gradient clipping (0 = disabled)
  
  # Learning rate scheduler (ReduceLROnPlateau)
  lr_factor: 0.5
  lr_patience: 5
  
  early_stopping: 100  # Epochs without improvement before stopping
```

### Run Management

```yaml
runs:
  base_dir: "./runs"
  auto_name: true          # Auto-generate timestamp-based names
  save_config: true        # Save config.yml with each run
  tensorboard_enabled: true
```

### Global Settings

```yaml
seed: 42      # Random seed for reproducibility
cpu: false    # Force CPU mode (true) or use GPU if available (false)
```

---

## Data Organization

### Directory Structure

```
data/
├── 6_sagittal_experiment/
│   ├── id.csv
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
├── 21_cross_section_control/
│   ├── id.csv
│   ├── image_001.tif
│   └── ...
├── 22_cross_section_treatment/
│   ├── id.csv
│   └── ...
└── metadata.csv
```

### id.csv Format

Each folder must contain an `id.csv` file mapping images to staging values:

```csv
filename,staging
image_001.tif,0.23
image_002.tif,0.45
image_003.tif,0.67
```

- **Column 1**: Filename (relative to folder)
- **Column 2**: Staging value (0-1 range, continuous)

The file is automatically sorted by staging value during loading.

### metadata.csv Format

The metadata file provides bit depth and PPM information for each folder:

```csv
folder_id,pixel_type,ppm
6,uint16,4.5
21,uint8,5.0
22,uint16,4.8
34,uint8,5.0
```

- **folder_id**: Numeric folder ID (extracted from folder name prefix)
- **pixel_type**: Image bit depth (uint8, uint16)
- **ppm**: Pixels per micron (used for normalization)

Missing PPM values are handled gracefully (image won't be scaled).

### Folder Naming Convention

Folders must start with a numeric ID followed by optional description:

- ✅ `6_sagittal_exp1`
- ✅ `21_control`
- ✅ `22_treatment_A`
- ❌ `experiment_6` (ID must be first)

---

## Architecture

### Image Processing Pipeline

The `CVImage` class implements a sophisticated 5-stage preprocessing pipeline:

#### Stage 1: Preprocessing

- **PPM Normalization**: Scales images to target pixels-per-micron ratio
- **Bit Depth Conversion**: Converts uint16 → uint8 or other formats to uint8
- **Resize & Pad**: Resizes to (img_width - 2×padding, img_height - 2×padding), then adds border padding

#### Stage 2: Segmentation

- **Low-Resolution Processing**: Segments at 256px for 4-16× speedup
- **Multi-Step Pipeline**:
  1. Normalization → Full intensity range
  2. Gaussian Blur → Noise reduction
  3. **Illumination Correction** → Flat-field correction for uneven lighting
  4. Otsu's Threshold → Binary segmentation
  5. Morphological Cleaning → Remove noise, fill gaps
  6. Flood Fill → Fill interior holes
  7. Largest Component → Extract embryo
  8. Border Expansion → Optional dilation
- **Upsampling**: Restores to original resolution

#### Stage 3: Border Detection

- **Contour Extraction**: Finds embryo outline
- **Multi-Stage Smoothing**:
  1. Uniform point distribution
  2. Coarse smoothing (10% kernel)
  3. Re-distribution to target points
  4. Fine smoothing (3-point kernel)
  5. Final uniform distribution

#### Stage 4: Border Extension

- **Normal Vectors**: Calculates inward/outward normals at each point
- **Inner/Outer Boundaries**:
  - `outward` (negative): Extends outward → outer edge (top surface)
  - `inward` (positive): Extends inward → inner edge (bottom surface)
- **Nuclear Layer Region**: Defines the ring between boundaries

#### Stage 5: Unrolling

- **Perspective Transform**: Segment-by-segment transformation
- **Flattening**: Converts curved nuclear layer to rectangular image
- **Top-Hat Filter**: Enhances contrast before unrolling
- **Output**: (depth × width) grayscale image ready for CNN

### Model Architecture

ResNet-style convolutional neural network for regression:

```
Input: (1, H, W) grayscale image
  ↓
Conv2d(1 → base_channels, 7×7, stride=2, padding=3)
BatchNorm2d + ReLU
MaxPool2d(3×3, stride=2)
  ↓
Layer 1: num_blocks[0] × ResidualBlock(base_channels)
  ↓
Layer 2: num_blocks[1] × ResidualBlock(base_channels × 2, stride=2)
  ↓
Layer 3: num_blocks[2] × ResidualBlock(base_channels × 4, stride=2)
  ↓
Layer 4: num_blocks[3] × ResidualBlock(base_channels × 8, stride=2)
  ↓
AdaptiveAvgPool2d(1×1)  # Handles variable width
  ↓
Flatten
Dropout(dropout_rate)
  ↓
Linear(base_channels × 8 → 1)
Sigmoid activation
  ↓
Output: Staging value ∈ [0, 1]
```

#### ResidualBlock

```
Input x
  ↓
├─> Conv2d(3×3, stride) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm ─┐
│                                                                       │
└─> Downsample (if needed) ────────────────────────────────────────────┤
                                                                        ↓
                                                                    Add + ReLU
                                                                        ↓
                                                                     Output
```

#### Model Variants

| Model  | Base Channels | Blocks     | Parameters | Use Case                    |
|--------|---------------|------------|------------|-----------------------------|
| Nano   | 16            | (1,1,1,1)  | ~50K       | Rapid prototyping           |
| Tiny   | 32            | (1,1,1,1)  | ~200K      | Limited compute             |
| Small  | 64            | (2,2,2,2)  | ~11M       | **Default**, good balance   |
| Medium | 96            | (2,2,2,2)  | ~25M       | More capacity               |
| Large  | 128           | (3,4,6,3)  | ~80M       | Maximum accuracy            |

---

## Project Structure

```
Staging_Model/
├── main.py                 # Main entry point
├── config.yml             # Configuration file
├── metadata.csv           # Bit depth and PPM metadata
├── pyproject.toml         # Package dependencies
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses
│   ├── cvimage.py         # Image processing pipeline (CVImage class)
│   ├── data.py            # Data loading and dataset classes
│   ├── functions.py       # Utility functions (border smoothing, normals)
│   ├── model.py           # ResNet model architecture
│   ├── run_manager.py     # Experiment tracking and run management
│   ├── tensorboard.py     # TensorBoard logging utilities
│   ├── test.py            # Testing and evaluation
│   ├── torchimage.py      # PyTorch image wrapper
│   └── train.py           # Training pipeline and trainer class
│
├── scripts/               # Utility scripts
│   └── process_images.py  # Batch image processing
│
├── tests/                 # Test scripts
│   ├── test_torch.py
│   ├── test_unroll_integration.py
│   └── verify_boundaries.py
│
├── runs/                  # Training runs (auto-generated)
│   └── 2026-01-16_18-30-45/
│       ├── config.yml
│       ├── checkpoints/
│       │   └── checkpoint_best.pt
│       └── logs/
│           └── events.out.tfevents...
│
├── output/                # Processed images (if enabled)
│
└── docs/                  # Documentation
```

### Key Modules

#### `src/config.py`

- Dataclasses for configuration management
- `AppConfig`: Top-level configuration
- `DataConfig`, `ModelConfig`, `TrainingConfig`: Sub-configurations
- Automatic YAML loading and validation

#### `src/cvimage.py`

- `CVImage`: Main image processing class
- Implements complete preprocessing pipeline
- Lazy computation with caching
- Methods: `get_original_image()`, `get_segmented_image()`, `get_nuclear_layer_image()`, `get_unrolled_image()`

#### `src/data.py`

- `Data`: Handles data loading from disk
- `TorchDataset`: PyTorch dataset with random interpolation augmentation
- Automatic train/val/test splitting

#### `src/model.py`

- `ResidualBlock`: Basic building block
- `Model`: Complete ResNet-style architecture
- `create_staging_model()`: Factory function for creating model variants

#### `src/train.py`

- `MetricsTracker`: Tracks MSE, MAE, RMSE, R² during training
- `Trainer`: Complete training loop with validation, checkpointing, and logging
- `train_model()`: High-level training function

#### `src/test.py`

- `test_model()`: Evaluation on test/val/custom datasets
- Generates scatter plots and computes metrics

#### `src/run_manager.py`

- `RunManager`: Manages training runs, checkpoints, and logs
- Auto-generates run names
- Creates directory structure

---

## Advanced Topics

### Run Management

The `RunManager` class automatically organizes experiments:

```
runs/
├── 2026-01-16_18-30-45/     # Auto-generated timestamp
│   ├── config.yml           # Configuration snapshot
│   ├── checkpoints/
│   │   └── checkpoint_best.pt
│   └── logs/
│       └── events.out.tfevents...
├── baseline_v1/             # Custom run name
│   └── ...
└── baseline_v2/
    └── ...
```

#### Access Run Information

```python
from src.run_manager import RunManager

run_mgr = RunManager(base_dir="./runs")

# Get latest run
latest = run_mgr.get_latest_run()

# Get run paths
checkpoint_dir = run_mgr.get_checkpoint_dir(latest)
logs_dir = run_mgr.get_logs_dir(latest)
```

### Checkpointing

Checkpoints are saved automatically during training:

#### Checkpoint Contents

```python
{
    'epoch': 42,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'train_metrics': {...},
    'val_metrics': {...},
    'best_val_loss': 0.0123,
    'config': {...}
}
```

#### Manual Checkpoint Loading

```python
import torch
from src.model import create_staging_model

# Load checkpoint
checkpoint = torch.load('checkpoint_best.pt')

# Create model
model = create_staging_model(
    model_type='small',
    dropout=0.5
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])

# Get training info
epoch = checkpoint['epoch']
best_loss = checkpoint['best_val_loss']
```

### Data Augmentation

The pipeline uses **random interpolation** for data augmentation:

```python
# In TorchDataset
def __getitem__(self, idx):
    folder = random.choice(self.folders)
    image_idx = random.randint(0, len(folder_data) - 2)
    
    # Load two adjacent images
    I1, id1 = load_image(image_idx)
    I2, id2 = load_image(image_idx + 1)
    
    # Random interpolation weight
    alpha = random.random()
    
    # Interpolate image and label
    I = (1 - alpha) * I1 + alpha * I2
    id_interp = (1 - alpha) * id1 + alpha * id2
    
    return I, id_interp
```

This creates effectively infinite training samples from discrete images.

### Custom Datasets

To train on your own data:

1. **Organize folders** with numeric IDs
2. **Create id.csv** in each folder
3. **Create metadata.csv** with PPM and bit depth info
4. **Update config.yml** with paths and folder IDs
5. **Run training**

#### Example Custom Configuration

```yaml
data:
  path: "/my/data/embryos"
  metadata_path: "/my/data/metadata.csv"
  test_ids: [101, 102]    # Your test folders
  val_ids: [103, 104]     # Your validation folders
  # Folders 105, 106, ... automatically used for training
  
  # Adjust for your microscope
  ppm: 3.2  # Your target pixels-per-micron
  
  # Adjust for your images
  img_height: 512
  img_width: 512
  
  # Fine-tune segmentation
  boundary_extension:
    cross_section:
      outward: -15
      inward: 30
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:

- Reduce `batch_size` in `config.yml`
- Use a smaller model: `model_type: "tiny"` or `"nano"`
- Reduce `img_height` and `img_width`
- Enable gradient accumulation (requires code modification)

#### 2. Segmentation Failures

**Symptoms**: Poor embryo detection, missing boundaries

**Solutions**:

- Check `plot_images: true` in code to visualize segmentation steps
- Adjust `boundary_extension` parameters
- Review illumination correction threshold (0.30 default)
- Ensure images have sufficient contrast

#### 3. No GPU Detected

**Symptoms**: Training very slow, `device: cpu` in logs

**Solutions**:

For NVIDIA GPUs:

- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Update GPU drivers
- Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

For Apple Silicon:

- Verify MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Ensure you're running on macOS 12.3+ with Apple Silicon (M1/M2/M3)
- If MPS is not available, the code will automatically fall back to CPU

#### 4. Poor Training Performance

**Symptoms**: High loss, low R² score

**Solutions**:

- Check learning rate: try `1e-6` to `1e-8`
- Ensure data is normalized correctly (0-1 range for staging)
- Increase model capacity: `model_type: "medium"` or `"large"`
- Increase training epochs
- Verify data quality and labels

#### 5. TensorBoard Not Found

**Symptoms**: `ModuleNotFoundError: No module named 'tensorboard'`

**Solution**:

```bash
uv pip install tensorboard
# or
pip install tensorboard
```

### Debugging Tips

#### Visualize Processing Steps

```python
from src.cvimage import CVImage
import cv2

# Enable visualization
cv_img = CVImage(
    I=image,
    id=0.5,
    plot_images=True,  # Show plots
    output_dir="./debug"  # Save images
)

# Process and inspect
unrolled = cv_img.get_unrolled_image()
```

#### Check Data Loading

```python
from src.data import Data

data = Data(
    path="/path/to/data",
    test=[21, 22],
    val=[6, 34],
    metadata_path="metadata.csv"
)

# Inspect splits
print(f"Train folders: {data.train_dir}")
print(f"Val folders: {data.val_dir}")
print(f"Test folders: {data.test_dir}")

# Get random sample
img, label, folder, idx = data.get_random_image('train')
print(f"Image shape: {img.shape}, Label: {label}")
```

#### Monitor GPU Usage

```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{staging_model_2026,
  author = {Your Name},
  title = {Staging Model for Embryo Nuclear Layer Analysis},
  year = {2026},
  url = {https://github.com/yourusername/staging-model}
}
```

---

## License

[Specify your license here]

---

## Contact

For questions, issues, or contributions:

- Email: <your.email@institution.edu>
- GitHub Issues: [https://github.com/yourusername/staging-model/issues](https://github.com/yourusername/staging-model/issues)

---

**Last Updated**: January 2026
