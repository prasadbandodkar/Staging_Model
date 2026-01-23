import yaml
from pathlib import Path
from dataclasses import dataclass, field
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
class ClassificationConfig:
    """Classification task configuration."""
    num_classes: int
    class_names: Optional[List[str]] = None

@dataclass
class TaskConfig:
    """
    Task configuration defining regression or classification.

    Attributes:
        type: Task type ('regression' or 'classification')
        classification: Classification-specific settings
    """
    type: Literal['regression', 'classification']
    classification: ClassificationConfig

@dataclass
class DataPaths:
    """Data path configuration."""
    root: str
    metadata: str

@dataclass
class DataSplits:
    """Dataset split configuration."""
    test_ids: List[int]
    val_ids: List[int]
    ignore_ids: List[int] = field(default_factory=list)

@dataclass
class DataLoading:
    """
    Data loading configuration.

    Controls how images are loaded from disk.
    """
    use_preprocessed: bool  # Load pre-processed images or run preprocessing on-the-fly
    trunc_width: Optional[int]  # Width truncation/cropping (applied during loading)
    num_workers: int  # Number of dataloader workers

@dataclass
class InterpolationAugmentation:
    """Interpolation augmentation settings."""
    enabled: bool
    distribution: str
    beta_alpha: float
    beta_beta: float

@dataclass
class DataAugmentation:
    """Data augmentation configuration."""
    enabled: bool
    interpolation: InterpolationAugmentation

@dataclass
class DataPreprocessing:
    """
    Image preprocessing configuration.

    Only used when loading.use_preprocessed=false.
    These parameters control the on-the-fly preprocessing pipeline.
    """
    img_height: int
    img_width: int
    padding: int
    image_type: Literal['original', 'segmented', 'nuclear_layer', 'unrolled']
    npoints: int
    boundary_extension: Dict[str, Dict[str, int]]
    sagittal_folder_prefixes: List[int]
    target_ppm: Optional[float]

@dataclass
class DataConfig:
    """
    Data configuration for image processing and dataset splitting.

    Organized by pipeline flow: paths → splits → loading → augmentation → preprocessing
    """
    paths: DataPaths
    splits: DataSplits
    loading: DataLoading
    augmentation: DataAugmentation
    preprocessing: DataPreprocessing

    def __post_init__(self):
        """Convert boundary_extension dicts to BoundaryExtension objects and validate configuration."""
        if isinstance(self.preprocessing.boundary_extension, dict):
            self.cross_section = BoundaryExtension(**self.preprocessing.boundary_extension['cross_section'])
            self.sagittal = BoundaryExtension(**self.preprocessing.boundary_extension['sagittal'])

        # Validate use_preprocessed configuration
        if self.loading.use_preprocessed and self.augmentation.interpolation.enabled:
            raise ValueError(
                "use_preprocessed=True requires augmentation.interpolation.enabled=False. "
                "When using pre-processed data, augmentation should already be baked into the dataset."
            )

    def get_boundary_params(self, folder_id: int) -> BoundaryExtension:
        """
        Get boundary extension parameters based on folder ID.

        Args:
            folder_id: Numeric folder ID

        Returns:
            BoundaryExtension object with inward/outward values
        """
        if folder_id in self.preprocessing.sagittal_folder_prefixes:
            return self.sagittal
        else:
            return self.cross_section

@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Attributes:
        architecture: Model architecture size to use.
                     Options: 'nano', 'tiny', 'small', 'medium', 'large'
        dropout: Dropout rate for regularization (0.0 to 1.0)
        show_summary: Whether to print a model summary at startup
    """
    architecture: str
    dropout: float
    show_summary: bool

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
class OptimizerConfig:
    """Optimizer configuration."""
    learning_rate: float
    weight_decay: float

@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str
    huber_delta: float

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str
    factor: float
    patience: int

@dataclass
class RegularizationConfig:
    """Regularization configuration."""
    grad_clip: float

@dataclass
class TrainingConfig:
    """
    Training hyperparameters and optimization settings.
    """
    optimizer: OptimizerConfig
    loss: LossConfig
    scheduler: SchedulerConfig
    regularization: RegularizationConfig
    batch_size: int
    epochs: int
    early_stopping: int
    checkpoint_dir: str

@dataclass
class SystemConfig:
    """System configuration."""
    seed: int
    device: str

@dataclass
class AppConfig:
    """
    Main application configuration containing all sub-configurations.

    Attributes:
        task: Task configuration (regression or classification)
        data: Data processing and dataset configuration
        model: Model architecture configuration
        training: Training hyperparameters
        runs: Run management and logging configuration
        system: System settings
    """
    task: TaskConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runs: RunsConfig
    system: SystemConfig

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
            task=TaskConfig(
                type=d['task']['type'],
                classification=ClassificationConfig(**d['task']['classification'])
            ),
            data=DataConfig(
                paths=DataPaths(**d['data']['paths']),
                splits=DataSplits(**d['data']['splits']),
                loading=DataLoading(**d['data']['loading']),
                augmentation=DataAugmentation(
                    enabled=d['data']['augmentation']['enabled'],
                    interpolation=InterpolationAugmentation(**d['data']['augmentation']['interpolation'])
                ),
                preprocessing=DataPreprocessing(**d['data']['preprocessing'])
            ),
            model=ModelConfig(**d['model']),
            training=TrainingConfig(
                optimizer=OptimizerConfig(**d['training']['optimizer']),
                loss=LossConfig(**d['training']['loss']),
                scheduler=SchedulerConfig(**d['training']['scheduler']),
                regularization=RegularizationConfig(**d['training']['regularization']),
                batch_size=d['training']['batch_size'],
                epochs=d['training']['epochs'],
                early_stopping=d['training']['early_stopping'],
                checkpoint_dir=d['training']['checkpoint_dir']
            ),
            runs=RunsConfig(**d['runs']),
            system=SystemConfig(**d['system'])
        )
