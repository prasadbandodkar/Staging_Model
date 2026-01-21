#
# PyTorch Dataset class for loading and preprocessing embryo images.
# This class combines data loading from the Data class with image processing
# (unrolling nuclear layer) and PyTorch Dataset interface for use with DataLoader.
#

# Main imports
from typing import Dict, List, Tuple, Optional, Literal

# Data handling imports
import pandas as pd
import numpy as np
import random

# Machine learning imports
import torch
from torch.utils.data import Dataset

# Local imports
from .data import Data, ListType
from .cvimage import CVImage
from .torchimage import TorchImage


class TorchDataset(Data, Dataset):
    """
    PyTorch Dataset for loading and preprocessing embryo images.

    Combines data loading from Data class with image processing (unrolling nuclear layer)
    and PyTorch Dataset interface for use with DataLoader.
    """

    def __init__(
        self,
        path: str,
        test: List[int] = [],
        val: List[int] = [],
        ignore: List[int] = [],
        size: Tuple[int, int] = (512, 512),
        padding: int = 44,
        npoints: int = 100,
        boundary_extension: Optional[Dict[str, Dict[str, int]]] = None,
        sagittal_folder_prefixes: Optional[List[int]] = None,
        trunc_width: Optional[int] = None,
        image_type: Literal['original', 'segmented', 'nuclear_layer', 'unrolled'] = 'unrolled',
        metadata_path: Optional[str] = None,
        target_ppm: float = 1.0,
        data_augment: bool = True,
        use_preprocessed: bool = False,
        augment_distribution: str = 'uniform',
        augment_beta_alpha: float = 0.5,
        augment_beta_beta: float = 0.5,
        task_type: str = 'regression',
        num_classes: int = 1,
        type: ListType = 'train'
    ) -> None:
        """
        Initialize the PyTorch Dataset.

        Args:
            path: Root path to the data directory
            test: List of folder IDs to use for testing
            val: List of folder IDs to use for validation
            ignore: List of folder IDs to ignore
            size: Target size for image resizing (height, width)
            padding: Padding to add around resized images
            npoints: Number of points for contour representation
            boundary_extension: Dict with cross_section and sagittal boundary params.
                               If None, uses default values for both.
            sagittal_folder_prefixes: List of folder IDs that are sagittal images.
                                     If None, defaults to [6, 7].
            trunc_width: Optional width to truncate unrolled images
            image_type: Type of image to use ('original', 'segmented', 'nuclear_layer', 'unrolled')
            metadata_path: Optional path to metadata CSV file
            target_ppm: Target pixels-per-micron for normalization (default: 1.0)
            data_augment: When True, randomly interpolate between adjacent images (training only).
                         When False, load images directly from disk without interpolation.
            use_preprocessed: When True, load pre-processed images directly, bypassing CVImage processing.
                             Requires data_augment=False and pre-processed data.
            augment_distribution: Distribution for interpolation sampling ('uniform' or 'beta')
            augment_beta_alpha: Alpha parameter for Beta distribution (only used if augment_distribution='beta')
            augment_beta_beta: Beta parameter for Beta distribution (only used if augment_distribution='beta')
            task_type: Task type ('regression' or 'classification')
            num_classes: Number of classes for classification (only used if task_type='classification')
            type: Dataset type to use ('train', 'test', or 'val')
        """
        super().__init__(
            path=path,
            test=test,
            val=val,
            ignore=ignore,
            metadata_path=metadata_path,
            augment_distribution=augment_distribution,
            augment_beta_alpha=augment_beta_alpha,
            augment_beta_beta=augment_beta_beta
        )
        self.size: Tuple[int, int] = size
        self.padding: int = padding
        self.npoints: int = npoints
        self.target_ppm: float = target_ppm

        # Parse boundary extension configuration
        if boundary_extension is None:
            # Default values if not provided
            boundary_extension = {
                'cross_section': {'inward': 34, 'outward': -30},
                'sagittal': {'inward': 34, 'outward': -30}
            }

        from .config import BoundaryExtension
        self.cross_section_params = BoundaryExtension(**boundary_extension['cross_section'])
        self.sagittal_params = BoundaryExtension(**boundary_extension['sagittal'])
        self.sagittal_folder_prefixes = sagittal_folder_prefixes if sagittal_folder_prefixes is not None else [6, 7]

        self.trunc_width: Optional[int] = trunc_width
        self.image_type: Literal['original', 'segmented', 'nuclear_layer', 'unrolled'] = image_type
        self.data_augment: bool = data_augment
        self.use_preprocessed: bool = use_preprocessed
        self.task_type: str = task_type
        self.num_classes: int = num_classes
        self.list_type: Optional[ListType] = None
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.indices: Optional[List[Tuple[str, int]]] = None
        self.type: ListType = type

        # Validate use_preprocessed configuration
        if self.use_preprocessed and self.data_augment:
            raise ValueError(
                "use_preprocessed=True requires data_augment=False. "
                "When using pre-processed data, augmentation should already be baked into the dataset."
            )

        self.set_list_type(type)

    def _get_boundary_params(self, folder: str) -> Tuple[int, int]:
        """
        Get boundary extension parameters based on folder type.

        Args:
            folder: Folder name (e.g., "6_somename")

        Returns:
            Tuple of (inward, outward) values
        """
        folder_id = self.get_folder_number(folder)

        if folder_id in self.sagittal_folder_prefixes:
            return (self.sagittal_params.inward, self.sagittal_params.outward)
        else:
            return (self.cross_section_params.inward, self.cross_section_params.outward)

    def _id_to_class(self, id_value: float) -> int:
        """
        Convert continuous ID value to discrete class label.

        Bins the ID value (assumed to be in range [0, 1]) into equal-width bins.

        Args:
            id_value: Continuous ID value in range [0, 1]

        Returns:
            Class label in range [0, num_classes-1]

        Examples:
            num_classes=2: [0.0, 0.5) → 0, [0.5, 1.0] → 1
            num_classes=5: [0.0, 0.2) → 0, [0.2, 0.4) → 1, ..., [0.8, 1.0] → 4
        """
        # Compute bin width
        bin_width = 1.0 / self.num_classes

        # Compute class index
        class_idx = int(id_value / bin_width)

        # Handle edge case: id_value == 1.0 should map to last class
        if class_idx >= self.num_classes:
            class_idx = self.num_classes - 1

        return class_idx

    def set_list_type(self, list_type: ListType) -> None:
        """
        Set the active dataset type and build the index list.

        Args:
            list_type: Type of dataset to use ('train', 'test', or 'val')

        Raises:
            ValueError: If list_type is not valid
        """
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")

        self.list_type = list_type
        self.data = list_dict[list_type]
        self.indices = [(folder, idx) for folder, df in self.data.items() for idx in range(len(df.index) - 1)]
        if list_type == 'train':
            random.shuffle(self.indices)
            # self.indices = [self.indices[i] for i in torch.randperm(len(self.indices))]

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            Number of samples available

        Raises:
            ValueError: If list_type has not been set
        """
        if self.indices is None:
            raise ValueError("List type not set. Call set_list_type() before using the dataset.")
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float, int]:
        """
        Get a processed image and its target at the specified index.

        Loads, interpolates, unrolls, and normalizes the image for training.

        Args:
            index: Index of the sample to retrieve

        Returns:
            For regression: Tuple of (normalized image tensor, continuous ID [0-1], folder ID)
            For classification: Tuple of (normalized image tensor, class label [0-num_classes], folder ID)

        Raises:
            ValueError: If list_type has not been set
        """
        if self.indices is None:
            raise ValueError("List type not set. Call set_list_type() before using the dataset.")

        folder, idx = self.indices[index]
        folder_id = self.get_folder_number(folder)

        if self.use_preprocessed:
            # Fast path: Load pre-processed (already unrolled) image directly
            I, id = self.get_raw_image(folder, idx, self.list_type)

            # Create TorchImage from the pre-processed image
            # TorchImage._prepare_tensor already normalizes to [0,1] range
            image = TorchImage(np.array(I, dtype=np.float32), id)

            # Apply truncation if needed (random crop for training)
            if self.trunc_width is not None and image.I.shape[2] > self.trunc_width:
                max_start = image.I.shape[2] - self.trunc_width
                if self.list_type == 'train':
                    # Random crop for training
                    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                else:
                    # Center crop for validation/test
                    start = max_start // 2
                image.I = image.I[:, :, start:start + self.trunc_width]

            # Apply augmentation for training data (operates on [0,1] range)
            if self.list_type == 'train':
                image.I = image.augment()

            # Convert target based on task type
            if self.task_type == 'classification':
                target = self._id_to_class(image.id)
            else:
                target = image.id

            return image.I, target, folder_id

        else:
            # Slow path: Load raw image and process through CVImage pipeline
            # Choose image loading method based on data_augment flag
            if self.data_augment:
                # Use random interpolation for data augmentation (original behavior)
                I, id = self.get_random_image_from_folder_idx(folder, idx, self.list_type)  # type: ignore
            else:
                # Load image directly from disk without interpolation
                I, id = self.get_raw_image(folder, idx, self.list_type)

            # Get folder-specific boundary parameters
            inward, outward = self._get_boundary_params(folder)

            # Get folder metadata for PPM scaling
            metadata = self.get_folder_metadata(folder_id)
            source_ppm = metadata.get('ppm', None)

            # Create CVImage instance with folder-specific parameters
            cv_image = CVImage(
                I=I,
                id=id,
                size=self.size,
                padding=self.padding,
                plot_images=False,
                npoints=self.npoints,
                inward=inward,
                outward=outward,
                trunc_width=self.trunc_width,
                source_ppm=source_ppm,
                target_ppm=self.target_ppm
            )

            # Get the image at the specified image type
            processed_image = cv_image.get_image(image_type=self.image_type, trunc_width=self.trunc_width)

            # Create TorchImage from the processed image
            # TorchImage._prepare_tensor already normalizes to [0,1] range
            image = TorchImage(np.array(processed_image, dtype=np.float32), id)

            # Apply augmentation for training data (operates on [0,1] range)
            if self.list_type == 'train':
                image.I = image.augment()

            # Convert target based on task type
            if self.task_type == 'classification':
                target = self._id_to_class(image.id)
            else:
                target = image.id

            # No need to normalize again - already in [0,1] from _prepare_tensor
            # If you need standardization (mean/std), add it here as a separate step
            return image.I, target, folder_id
