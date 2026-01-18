import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Literal

@dataclass
class BoundaryExtension:
    """
    Boundary extension configuration for nuclear layer.
    
    Attributes:
        inward: Distance to extend inward (toward center) → inner edge (bottom surface) of nuclear layer.
                Use positive values to extend toward the embryo center.
        outward: Distance to extend outward (away from center) → outer edge (top surface) of nuclear layer.
                 Use negative values to extend away from the detected border.
    """
    inward: int
    outward: int

@dataclass
class DataConfig:
    """
    Data configuration for image processing and dataset splitting.
    
    Attributes:
        path: Path to the directory containing the dataset
        metadata_path: Path to metadata CSV file with bit depth and PPM information
        test_ids: Folder IDs associated with the test set (datasets usually split by folder ID)
        val_ids: Folder IDs associated with the validation set
        ppm: Target pixels-per-micron for image normalization (1.0 = 1 pixel = 1 micron)
        img_height: Target image height after preprocessing (images will be resized to this height)
        img_width: Target image width after preprocessing (images will be resized to this width)
        padding: Padding to add around the resized image during preprocessing
        npoints: Number of points used for contour representation
        boundary_extension: Nested configuration for cross-section and sagittal boundary extension.
                           The nuclear layer is located at the OUTER surface of the embryo.
                           The detected border is approximately in the middle of this nuclear layer.
        sagittal_folder_prefixes: List of folder prefixes that contain sagittal images.
                                 All other folders are assumed to be cross-section.
        trunc_width: Optional width truncation for training (None to disable)
        image_type: Type of image to use for training.
                   Options: 'original', 'segmented', 'nuclear_layer', 'unrolled'
                   - 'original': Original input image resized but not padded
                   - 'segmented': Binary segmentation mask of entire embryo
                   - 'nuclear_layer': Binary mask of nuclear layer region (not unrolled)
                   - 'unrolled': Unrolled nuclear layer with optional width truncation (recommended)
        data_augment: When true, randomly interpolate between adjacent images during training.
                     When false, load images directly from disk without interpolation.
        num_workers: Number of data loading workers (typically set to number of CPU cores, or 0 for single-threaded)
    """
    path: str
    metadata_path: str
    test_ids: List[int]
    val_ids: List[int]
    ppm: Optional[float]
    img_height: int 
    img_width: int
    padding: int
    npoints: int
    boundary_extension: Dict[str, Dict[str, int]]  # Will be converted to BoundaryExtension objects
    sagittal_folder_prefixes: List[int]
    trunc_width: Optional[int]
    image_type: Literal['original', 'segmented', 'nuclear_layer', 'unrolled']
    data_augment: bool
    num_workers: int
    
    def __post_init__(self):
        """Convert boundary_extension dicts to BoundaryExtension objects."""
        if isinstance(self.boundary_extension, dict):
            self.cross_section = BoundaryExtension(**self.boundary_extension['cross_section'])
            self.sagittal = BoundaryExtension(**self.boundary_extension['sagittal'])
    
    def get_boundary_params(self, folder_id: int) -> BoundaryExtension:
        """
        Get boundary extension parameters based on folder ID.
        
        Args:
            folder_id: Numeric folder ID
            
        Returns:
            BoundaryExtension object with inward/outward values
        """
        if folder_id in self.sagittal_folder_prefixes:
            return self.sagittal
        else:
            return self.cross_section

@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Attributes:
        model_type: Model architecture size to use.
                   Options: 'nano', 'tiny', 'small', 'medium', 'large'
                   - nano:   <150k params
                   - tiny:   ~700k params
                   - small:  ~11M params (ResNet18-like)
                   - medium: ~25M params (ResNet34-like)
                   - large:  ~45M params
        dropout: Dropout rate for regularization (0.0 to 1.0)
        summary: Whether to print a model summary at startup
    """
    model_type: str
    dropout: float
    summary: bool

@dataclass
class RunsConfig:
    """
    Configuration for training run management and logging.
    
    Attributes:
        base_dir: Base directory for all training runs
        auto_name: Auto-generate timestamp-based run names
        save_config: Save config.yml snapshot in run directory
        tensorboard_enabled: Enable TensorBoard logging
    """
    base_dir: str
    auto_name: bool
    save_config: bool
    tensorboard_enabled: bool


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and optimization settings.
    
    Attributes:
        batch_size: Batch size for training and validation
        epochs: Total number of training epochs
        learning_rate: Learning rate for the optimizer
        weight_decay: L2 regularization strength (weight decay)
        loss_type: Loss function to use. Options: 'mse' (Mean Squared Error), 'smooth_l1', 'huber'
        huber_delta: Delta parameter for Huber loss (only used if loss_type is 'huber')
        grad_clip: Gradient clipping norm (set to 0 to disable)
        lr_factor: Learning rate reduction factor (multiply LR by this factor when reducing)
        lr_patience: Number of epochs with no improvement before reducing LR
        early_stopping: Stop training if validation loss doesn't improve for this many epochs
        checkpoint_dir: Directory where model checkpoints will be saved
    """
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    loss_type: str
    huber_delta: float
    grad_clip: float
    lr_factor: float
    lr_patience: int
    early_stopping: int
    checkpoint_dir: str

@dataclass
class AppConfig:
    """
    Main application configuration containing all sub-configurations.
    
    Attributes:
        data: Data processing and dataset configuration
        model: Model architecture configuration
        training: Training hyperparameters
        runs: Run management and logging configuration
        seed: Random seed for reproducibility
        cpu: Force CPU usage even if CUDA is available
    """
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runs: RunsConfig
    seed: int
    cpu: bool

    @classmethod
    def load(cls, config_path: str = "config.yml") -> "AppConfig":
        """Load configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            # Try looking in parent directory if we are in src/
            parent_path = Path("..") / config_path
            if parent_path.exists():
                path = parent_path
            else:
                raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
            
        return cls.from_dict(cfg_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        """Create config object from dictionary."""
        return cls(
            data=DataConfig(**d['data']),
            model=ModelConfig(**d['model']),
            training=TrainingConfig(**d['training']),
            runs=RunsConfig(**d['runs']),
            seed=d['seed'],
            cpu=d['cpu']
        )
