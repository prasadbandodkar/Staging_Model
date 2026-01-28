#
# This is a class to handle the files on the disc. The idea is to load a file from the id.csv file, and 
# use it to obtain synthetic images that are interpolated from two adjacent images. This class will also
# handle regression ids
#

# Main imports
import os
import sys
import csv
from typing import Dict, List, Tuple, Optional, Literal

# Data handling imports
import pandas as pd
import numpy as np
import numpy.typing as npt
import cv2 as cv

# Machine learning imports
import torch

# Type hints
ListType = Literal['train', 'test', 'val']

class Data:
    """
    Handles data loading and management from disk for training, validation, and testing.
    
    This class loads images from folders organized by ID, with support for interpolation
    between adjacent imag∏es to generate synthetic training data.
    """
    
    def __init__(self,
                 path: str,
                 test: List[int] = [],
                 val: List[int] = [],
                 ignore: List[int] = [],
                 metadata_path: Optional[str] = None,
                 augment_distribution: Optional[str] = 'uniform',
                 augment_beta_alpha: Optional[float] = 0.5,
                 augment_beta_beta: Optional[float] = 0.5) -> None:
        """
        Initialize the Data handler.

        Args:
            path: Root path to the data directory
            test: List of folder IDs to use for testing
            val: List of folder IDs to use for validation
            ignore: List of folder IDs to ignore
            metadata_path: Optional path to metadata CSV file
            augment_distribution: Distribution for interpolation sampling ('uniform' or 'beta')
            augment_beta_alpha: Alpha parameter for Beta distribution (only used if augment_distribution='beta')
            augment_beta_beta: Beta parameter for Beta distribution (only used if augment_distribution='beta')
        """
        self.data_path  : str       = path
        self.train_list : List[str] = []
        self.val_list   : List[str] = []
        self.test_list  : List[str] = []
        self.ignore_list: List[str] = []

        self.train_data: Dict[str, pd.DataFrame] = {}
        self.val_data  : Dict[str, pd.DataFrame] = {}
        self.test_data : Dict[str, pd.DataFrame] = {}

        # Cache for valid interpolation pairs (folder -> list of valid idx pairs)
        self._valid_pairs_cache: Dict[str, List[int]] = {}

        # Augmentation configuration
        self.augment_distribution = augment_distribution
        self.augment_beta_alpha = augment_beta_alpha
        self.augment_beta_beta = augment_beta_beta

        # Load metadata if provided
        self.metadata: Dict[int, Dict[str, any]] = {}
        if metadata_path:
            self._load_metadata(metadata_path)

        self.train_test_val_data(test, val, ignore)


    def train_test_val_dir(self, test: List[int], val: List[int], ignore: List[int] = []) -> None:
        """
        Split folders into train, test, validation, and ignore lists based on folder IDs.
        
        Args:
            test: List of folder IDs to use for testing
            val: List of folder IDs to use for validation
            ignore: List of folder IDs to ignore
        """
        folders = os.listdir(self.data_path)
        for folder in folders:
            if not os.path.isdir(os.path.join(self.data_path, folder)):
                continue                        # Skip files
            number_before_underscore = folder.split('_')[0]
            if not number_before_underscore.isdigit():
                continue                        # Skip if the character before the underscore is not a number
            number_before_underscore = int(number_before_underscore)
            if number_before_underscore in test:
                self.test_list.append(folder)
            elif number_before_underscore in val:
                self.val_list.append(folder)
            elif number_before_underscore in ignore:
                self.ignore_list.append(folder)
            else:
                self.train_list.append(folder)      # default is train
    
    
    def get_folder_data(self, folders: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Reads the 'id.csv' file from each specified folder, sorts it by the second column, 
        and appends the data path to the file name in the first column. 

        Args:
            folders (list): A list of folder names.

        Returns:
            dict: A dictionary where the keys are the folder names and the values are 
            pandas DataFrames containing the sorted data from the 'id.csv' files in each folder.
        """
        data: Dict[str, pd.DataFrame] = {}
        for folder in folders:
            id_file = os.path.join(self.data_path, folder, "id.csv")
            df = pd.read_csv(id_file, header=None)
            df = df.sort_values(by=1)  # type: ignore       # unfortunately the csv file is not sorted. We need to sort it here.
            # add self.data_path to the file name - the first column
            df[0] = df[0].apply(lambda x: os.path.join(self.data_path, x))
            data[folder] = df
        return data
    

    def train_test_val_data(self, test: List[int] = [], val: List[int] = [], ignore: List[int] = []) -> None:
        """
        Load and organize data from train, test, and validation folders.
        
        Args:
            test: List of folder IDs to use for testing
            val: List of folder IDs to use for validation
            ignore: List of folder IDs to ignore
        """
        if not self.train_list or not self.val_list or not self.test_list:
            self.train_test_val_dir(test, val, ignore)
        self.train_data = self.get_folder_data(self.train_list)
        self.val_data   = self.get_folder_data(self.val_list)
        self.test_data  = self.get_folder_data(self.test_list)
    
    
    def get_folder_number(self, folder: str) -> int:
        """
        Extract the numeric ID from a folder name.
        
        Args:
            folder: Folder name in format "ID_name"
            
        Returns:
            The numeric ID from the folder name
            
        Raises:
            ValueError: If folder name doesn't start with a digit
        """
        number = folder.split('_')[0]
        # check if the number is a digit
        if not number.isdigit():
            raise ValueError(f"Folder name '{folder}' is not in the correct format.")
        return int(number)
   
   
    def _load_metadata(self, metadata_path: str) -> None:
        """
        Load metadata from CSV file and index by folder ID.
        
        Args:
            metadata_path: Path to metadata CSV file
        """
        try:
            with open(metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract folder ID from filename (e.g., "1_1_Embryo..." -> 1)
                    filename = row['Filename']
                    if not filename:
                        continue
                    
                    folder_id_str = filename.split('_')[0]
                    try:
                        folder_id = int(folder_id_str)
                    except ValueError:
                        continue
                    
                    #Parse PPM (may be empty string)
                    ppm_str = row.get('ppm', '').strip()
                    ppm = float(ppm_str) if ppm_str else None
                    
                    # Store metadata
                    self.metadata[folder_id] = {
                        'filename': filename,
                        'pixel_type': row.get('Pixel Type', 'uint16'),
                        'ppm': ppm
                    }
                    
            print(f"Loaded metadata for {len(self.metadata)} folders")
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
            self.metadata = {}
    
    
    def get_folder_metadata(self, folder_id: int) -> Dict[str, any]:
        """
        Get metadata for a specific folder ID.
        
        Args:
            folder_id: The folder ID
            
        Returns:
            Dictionary with keys: 'pixel_type', 'ppm' (may be None if missing)
        """
        if folder_id in self.metadata:
            return self.metadata[folder_id]
        else:
            # Return defaults if metadata not found
            return {
                'filename': None,
                'pixel_type': 'uint16',
                'ppm': None
            }
    
    
    def _parse_filename_metadata(self, filename: str) -> Optional[Dict[str, int]]:
        """
        Parse filename to extract s, c, z, t metadata.
        
        Args:
            filename: Full path to image file (e.g., ".../s1_c2_z1_t38_orig.png" or ".../s1_c2_z1_t38_aug_0.png")
            
        Returns:
            Dictionary with 's', 'c', 'z', 't' keys, or None if parsing fails
        """
        try:
            # Extract just the filename without path and extension
            basename = os.path.basename(filename)
            name_without_ext = os.path.splitext(basename)[0]
            
            # Parse s, c, z, t values (format: s1_c2_z1_t38_orig or s1_c2_z1_t38_aug_0)
            # We only need the first 4 parts (s, c, z, t), ignore the rest (_orig, _aug_0, etc.)
            parts = name_without_ext.split('_')
            if len(parts) < 4:
                return None
                
            metadata = {}
            # Only parse first 4 parts
            for part in parts[:4]:
                if len(part) < 2:
                    return None
                key = part[0]  # 's', 'c', 'z', or 't'
                value_str = part[1:]  # the numeric part
                
                if key not in ['s', 'c', 'z', 't'] or not value_str.isdigit():
                    return None
                    
                metadata[key] = int(value_str)
            
            return metadata
        except Exception:
            return None
    
    
    def _build_valid_interpolation_pairs(self, folder: str, list_type: ListType) -> List[int]:
        """
        Build a list of valid starting indices for interpolation.
        
        Two consecutive images at idx and idx+1 are valid for interpolation if:
        - They have the same s, c, z values
        - They have consecutive t values (t2 = t1 + 1)
        
        Args:
            folder: Name of the folder to analyze
            list_type: Type of dataset ('train', 'test', or 'val')
            
        Returns:
            List of valid starting indices for interpolation
        """
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        
        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")
        
        data = list_dict[list_type]
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in data.")
        
        df = data[folder]
        valid_pairs = []
        
        # Check each consecutive pair
        for idx in range(len(df) - 1):
            filename1 = df.iloc[idx, 0]
            filename2 = df.iloc[idx + 1, 0]
            
            meta1 = self._parse_filename_metadata(filename1)
            meta2 = self._parse_filename_metadata(filename2)
            
            # Skip if parsing failed
            if meta1 is None or meta2 is None:
                continue
            
            # Check if s, c, z match and t values are consecutive
            if (meta1['s'] == meta2['s'] and 
                meta1['c'] == meta2['c'] and 
                meta1['z'] == meta2['z'] and 
                meta2['t'] == meta1['t'] + 1):
                valid_pairs.append(idx)
        
        return valid_pairs
    
    
    def get_raw_image(self, folder: str, idx: int, list_type: ListType) -> Tuple[npt.NDArray[np.uint8], float]:
        """
        Load a raw image from disk by folder name, index, and list type.
        
        Args:
            folder: Name of the folder containing the image
            idx: Index of the image within the folder
            list_type: Type of dataset ('train', 'test', or 'val')
            
        Returns:
            Tuple of (image array, image ID)
            
        Raises:
            ValueError: If list_type is invalid
            KeyError: If folder is not found
            IndexError: If index is out of bounds
            FileNotFoundError: If image file cannot be loaded
        """
        # Check if list_type is valid
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError(f"Invalid list_type '{list_type}'. Expected one of: {list(list_dict.keys())}")

        data = list_dict[list_type]

        # Check if folder is valid
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in {list_type} data.")

        # Check if idx is valid
        if not (0 <= idx < len(data[folder])):
            raise IndexError(f"Index {idx} is out of bounds for folder '{folder}' with size {len(data[folder])}.")

        filename: str = data[folder].iloc[idx, 0]
        id: float = data[folder].iloc[idx, 1]
        I = cv.imread(filename, cv.IMREAD_GRAYSCALE)

        # Check if the image was successfully loaded
        if I is None:
            raise FileNotFoundError(f"Image '{filename}' could not be loaded.")

        return I, id
       
            
    def get_raw_image_old(self, folder: str, idx: int, list_type: ListType) -> Tuple[npt.NDArray[np.uint8], float]:
        """
        Legacy method to load a raw image (without validation).
        
        Args:
            folder: Name of the folder containing the image
            idx: Index of the image within the folder
            list_type: Type of dataset ('train', 'test', or 'val')
            
        Returns:
            Tuple of (image array, image ID)
        """
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        data = list_dict[list_type]
        filename: str = data[folder].iloc[idx, 0]
        id: float = data[folder].iloc[idx, 1]
        I = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        return I, id
    
    
    
    def get_random_image(self, list_type: ListType) -> Tuple[npt.NDArray[np.uint8], float, str, int]:
        """
        Get a random interpolated image from the specified dataset.

        Args:
            list_type: The type of list to get the image from ('train', 'test', or 'val')

        Returns:
            Tuple of (image array, id, folder name, image index)
        """
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }

        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")

        data = list_dict[list_type]
        folders = list(data.keys())
        rfolder_idx: int = torch.randint(len(folders), (1,)).item()
        rfolder: str = folders[rfolder_idx]
        
        I, id, idx = self.get_random_image_from_folder(rfolder, list_type)
        
        return I, id, rfolder, idx


    def get_random_image_from_folder(self, folder: str, list_type: ListType) -> Tuple[npt.NDArray[np.uint8], float, int]:
        """
        Get a random interpolated image from a specific folder.

        For training: Ensures interpolation only happens between images with matching
        s, c, z values and consecutive t values.
        For test/val: Returns any random image (no interpolation).

        Args:
            folder: Name of the folder to sample from
            list_type: The type of list to get the image from ('train', 'test', or 'val')

        Returns:
            Tuple of (image array, id, image index)
        """
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")

        data = list_dict[list_type]
        
        # Check if folder is not in data and raise an error if true
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in data.")
        
        # For training, use valid interpolation pairs
        if list_type == 'train':
            # Build cache key
            cache_key = f"{list_type}_{folder}"
            
            # Build valid pairs if not cached
            if cache_key not in self._valid_pairs_cache:
                valid_pairs = self._build_valid_interpolation_pairs(folder, list_type)
                if not valid_pairs:
                    raise ValueError(f"No valid interpolation pairs found in folder '{folder}'. "
                                   "Images must have matching s, c, z values with consecutive t values.")
                self._valid_pairs_cache[cache_key] = valid_pairs
            
            # Sample from valid pairs
            valid_pairs = self._valid_pairs_cache[cache_key]
            random_pair_idx = torch.randint(len(valid_pairs), (1,)).item()
            idx = valid_pairs[random_pair_idx]
        else:
            # For test/val, use any index (no interpolation anyway)
            idx: int = torch.randint(len(data[folder].index), (1,)).item()
        
        I, id = self.get_random_image_from_folder_idx(folder, idx, list_type)
        
        return I, id, idx
      

    def get_random_image_from_folder_idx(self, folder: str, idx: int, list_type: ListType) -> Tuple[npt.NDArray[np.uint8], float]:
        """
        Get an image from a specific folder and index.

        For training: Creates a synthetic image by randomly interpolating between two adjacent images
        ONLY if they form a valid pair (matching s, c, z with consecutive t values).
        For test/val: Returns the actual raw image without interpolation for consistent evaluation.

        Args:
            folder: Name of the folder containing the images
            idx: Starting index (for training, will use idx and idx+1 for interpolation if valid)
            list_type: The type of list to get images from ('train', 'test', or 'val')

        Returns:
            Tuple of (image array, id)
            
        Raises:
            ValueError: If idx is not a valid interpolation pair for training data
        """
        I1, id1 = self.get_raw_image(folder, idx, list_type)

        # Only interpolate for training data AND if it's a valid pair
        if list_type == 'train':
            # Check if this idx forms a valid interpolation pair
            list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
                'train': self.train_data,
                'test': self.test_data,
                'val': self.val_data
            }
            data = list_dict[list_type]
            
            # Validate the pair
            if idx >= len(data[folder]) - 1:
                # Can't interpolate: no next image
                raise ValueError(f"Index {idx} cannot be used for interpolation (no next image)")
            
            filename1 = data[folder].iloc[idx, 0]
            filename2 = data[folder].iloc[idx + 1, 0]
            
            meta1 = self._parse_filename_metadata(filename1)
            meta2 = self._parse_filename_metadata(filename2)
            
            # Check if this is a valid pair for interpolation
            is_valid_pair = (
                meta1 is not None and 
                meta2 is not None and
                meta1['s'] == meta2['s'] and 
                meta1['c'] == meta2['c'] and 
                meta1['z'] == meta2['z'] and 
                meta2['t'] == meta1['t'] + 1
            )
            
            if not is_valid_pair:
                # Not a valid pair - return original image without interpolation
                # This prevents incorrect augmentation when pairs don't match
                return I1, id1
            
            # Valid pair - proceed with interpolation
            I2, id2 = self.get_raw_image(folder, idx+1, list_type)

            # Sample alpha based on configured distribution
            if self.augment_distribution == 'beta':
                # Beta distribution: Beta(alpha, beta)
                # Beta(0.5, 0.5) creates U-shape favoring endpoints (original images)
                # Beta(1, 1) is uniform distribution
                # Beta(2, 2) favors center (more interpolation)
                alpha: float = torch.distributions.Beta(self.augment_beta_alpha, self.augment_beta_beta).sample().item()
            else:
                # Uniform distribution [0, 1] - default
                # Equal probability for all interpolation values
                alpha: float = torch.rand(1).item()

            I = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
            id: float = alpha*id1 + (1-alpha)*id2
        else:
            # Test/val: use actual raw images (no interpolation)
            I, id = I1, id1

        return I, id



    def get_augmented_images_from_folder_idx(
        self, 
        folder: str, 
        idx: int, 
        num_augmentations: int,
        list_type: ListType = 'train'
    ) -> List[Tuple[npt.NDArray[np.uint8], float]]:
        """
        Generate multiple interpolated images using a uniform grid of alpha values.
        
        This function is designed for data preparation where we want deterministic,
        evenly-spaced augmentations. Unlike get_random_image_from_folder_idx which
        uses random sampling, this creates num_augmentations images with alpha values
        evenly distributed between 0 and 1.
        
        Only creates interpolated images if idx and idx+1 form a valid pair
        (matching s, c, z with consecutive t values).
        
        Args:
            folder: Name of the folder containing the images
            idx: Starting index (will use idx and idx+1 for interpolation if valid)
            num_augmentations: Number of augmented images to generate
            list_type: Type of dataset ('train', 'test', or 'val')
            
        Returns:
            List of tuples (image array, id) for each augmented image.
            If not a valid pair, returns empty list.
            
        Example:
            For num_augmentations=3, creates images with alpha=[0.25, 0.5, 0.75]
            (evenly spaced between 0 and 1, excluding endpoints)
        """
        # Check if this idx forms a valid interpolation pair
        list_dict: Dict[str, Dict[str, pd.DataFrame]] = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        
        if list_type not in list_dict:
            raise ValueError("list_type must be 'train', 'test', or 'val'")
        
        data = list_dict[list_type]
        
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in {list_type} data.")
        
        # Validate the pair
        if idx >= len(data[folder]) - 1:
            # Can't interpolate: no next image
            return []
        
        filename1 = data[folder].iloc[idx, 0]
        filename2 = data[folder].iloc[idx + 1, 0]
        
        meta1 = self._parse_filename_metadata(filename1)
        meta2 = self._parse_filename_metadata(filename2)
        
        # Check if this is a valid pair for interpolation
        is_valid_pair = (
            meta1 is not None and 
            meta2 is not None and
            meta1['s'] == meta2['s'] and 
            meta1['c'] == meta2['c'] and 
            meta1['z'] == meta2['z'] and 
            meta2['t'] == meta1['t'] + 1
        )
        
        if not is_valid_pair:
            # Not a valid pair - return empty list
            return []
        
        # Load both images
        I1, id1 = self.get_raw_image(folder, idx, list_type)
        I2, id2 = self.get_raw_image(folder, idx + 1, list_type)
        
        # Generate uniform grid of alpha values
        # Exclude 0 and 1 to avoid duplicating original images
        # For num_augmentations=3: alpha = [0.25, 0.5, 0.75]
        # For num_augmentations=2: alpha = [0.33, 0.67]
        alpha_values = np.linspace(0, 1, num_augmentations + 2)[1:-1]
        
        # Generate augmented images
        augmented_images = []
        for alpha in alpha_values:
            # Interpolate image and ID
            I_aug = cv.addWeighted(I1, alpha, I2, 1 - alpha, 0)
            id_aug = alpha * id1 + (1 - alpha) * id2
            augmented_images.append((I_aug, id_aug))
        
        return augmented_images


if __name__ == "__main__":
    path = "/Volumes/X2/Projects/staging/Data/data"
    f = Data(path)


    # create train, test, and val
    test = [6,7]
    val  = [21, 34]
    f.train_test_val_dir(test, val)
    # print test_list
    # print(f.test_list)
    # print(f.val_list)
    # print(f.train_list)
    
    
    # get folder data
    f.train_test_val_data()
    
    f.get_random_image('train')
    
    
    
