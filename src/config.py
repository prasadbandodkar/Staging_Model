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
    use_unroll_on_fly: bool  # When False: load pre-processed unrolled images, when True: unroll on-the-fly
    trunc_width: Optional[int]  # Width truncation/cropping (applied during loading)
    num_workers: int  # Number of dataloader workers

@dataclass
class UnrollOnFlyConfig:
    """
    Unroll on-the-fly configuration.

    Only used when loading.use_unroll_on_fly=true.
    This pipeline always produces unrolled images.
    """
    img_height: int
    img_width: int
    padding: int
    npoints: int
    boundary_extension: Dict[str, Dict[str, int]]
    sagittal_folder_prefixes: List[int]
    target_ppm: Optional[float]
    augmentation_enabled: bool
    interpolation_enabled: bool
    interpolation_distribution: str
    interpolation_beta_alpha: float
    interpolation_beta_beta: float

@dataclass
class DataConfig:
    """
    Data configuration for image processing and dataset splitting.

    Organized by pipeline flow: paths → splits → image_type → loading → unroll_on_fly
    """
    paths: DataPaths
    splits: DataSplits
    image_type: Literal['unrolled', 'original']
    loading: DataLoading
    unroll_on_fly: UnrollOnFlyConfig

    def __post_init__(self):
        """Convert boundary_extension dicts to BoundaryExtension objects and validate configuration."""
        if isinstance(self.unroll_on_fly.boundary_extension, dict):
            self.cross_section = BoundaryExtension(**self.unroll_on_fly.boundary_extension['cross_section'])
            self.sagittal = BoundaryExtension(**self.unroll_on_fly.boundary_extension['sagittal'])

        # Validate use_unroll_on_fly configuration
        if not self.loading.use_unroll_on_fly and self.unroll_on_fly.interpolation_enabled:
            raise ValueError(
                "use_unroll_on_fly=False requires interpolation_enabled=False. "
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
        if folder_id in self.unroll_on_fly.sagittal_folder_prefixes:
            return self.sagittal
        else:
            return self.cross_section

@dataclass
class NormalizationConfig:
    """
    Normalization layer configuration.
    
    Attributes:
        type: Type of normalization ('group_norm' or 'batch_norm')
        num_groups: Number of groups for GroupNorm (only used if type='group_norm')
    """
    type: Literal['group_norm', 'batch_norm']
    num_groups: int = 32

@dataclass
class SEBlockConfig:
    """
    Squeeze-and-Excitation block configuration.
    
    Attributes:
        enabled: Whether to use SE blocks for channel attention
        reduction: Reduction ratio for the SE bottleneck (higher = fewer params)
    """
    enabled: bool
    reduction: int = 16

@dataclass
class ResidualBlockConfig:
    """
    Residual block design configuration.
    
    Attributes:
        pre_activation: Use pre-activation design (BN→Activation→Conv) vs post-activation
    """
    pre_activation: bool

@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Attributes:
        architecture: Model architecture size to use.
                     Options: 'nano', 'tiny', 'small', 'medium', 'large'
        dropout: Dropout rate for regularization (0.0 to 1.0)
        show_summary: Whether to print a model summary at startup
        activation: Activation function to use ('gelu', 'relu', 'mish', 'leaky_relu')
        normalization: Normalization layer configuration
        se_block: Squeeze-and-Excitation block configuration
        residual_block: Residual block design configuration
    """
    architecture: str
    dropout: float
    show_summary: bool
    activation: Literal['gelu', 'relu', 'mish', 'leaky_relu'] = 'gelu'
    normalization: NormalizationConfig = field(default_factory=lambda: NormalizationConfig(type='group_norm', num_groups=32))
    se_block: SEBlockConfig = field(default_factory=lambda: SEBlockConfig(enabled=True, reduction=16))
    residual_block: ResidualBlockConfig = field(default_factory=lambda: ResidualBlockConfig(pre_activation=True))

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
class DropoutScheduleConfig:
    """Dropout schedule configuration."""
    type: str  # constant, linear, step
    dropout_start: float = 0.5
    dropout_end: float = 0.5
    step_epochs: List[int] = field(default_factory=list)
    step_values: List[float] = field(default_factory=list)

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str
    factor: float
    patience: int
    warmup_epochs: int = 0

@dataclass
class RegularizationConfig:
    """Regularization configuration."""
    grad_clip: float
    dropout_schedule: DropoutScheduleConfig

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
                image_type=d['data']['image_type'],
                loading=DataLoading(**d['data']['loading']),
                unroll_on_fly=UnrollOnFlyConfig(**d['data']['unroll_on_fly'])
            ),
            model=ModelConfig(
                architecture=d['model']['architecture'],
                dropout=d['model']['dropout'],
                show_summary=d['model']['show_summary'],
                activation=d['model'].get('activation', 'gelu'),
                normalization=NormalizationConfig(**d['model'].get('normalization', {'type': 'group_norm', 'num_groups': 32})),
                se_block=SEBlockConfig(**d['model'].get('se_block', {'enabled': True, 'reduction': 16})),
                residual_block=ResidualBlockConfig(**d['model'].get('residual_block', {'pre_activation': True}))
            ),
            training=TrainingConfig(
                optimizer=OptimizerConfig(**d['training']['optimizer']),
                loss=LossConfig(**d['training']['loss']),
                scheduler=SchedulerConfig(**d['training']['scheduler']),
                regularization=RegularizationConfig(
                    grad_clip=d['training']['regularization']['grad_clip'],
                    dropout_schedule=DropoutScheduleConfig(**d['training']['regularization']['dropout_schedule'])
                ),
                batch_size=d['training']['batch_size'],
                epochs=d['training']['epochs'],
                early_stopping=d['training']['early_stopping'],
                checkpoint_dir=d['training']['checkpoint_dir']
            ),
            runs=RunsConfig(**d['runs']),
            system=SystemConfig(**d['system'])
        )
