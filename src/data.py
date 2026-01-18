#
# This is a class to handle the files on the disc. The idea is to load a file from the id.csv file, and 
# use it to obtain synthetic images that are interpolated from two adjacent images. This class will also
# handle regression ids
#

# Main imports
import os
import sys
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
                 metadata_path: Optional[str] = None) -> None:
        """
        Initialize the Data handler.
        
        Args:
            path: Root path to the data directory
            test: List of folder IDs to use for testing
            val: List of folder IDs to use for validation
            ignore: List of folder IDs to ignore
            metadata_path: Optional path to metadata CSV file
        """
        self.data_path  : str       = path
        self.train_list : List[str] = []
        self.val_list   : List[str] = []
        self.test_list  : List[str] = []
        self.ignore_list: List[str] = []
        
        self.train_data: Dict[str, pd.DataFrame] = {}
        self.val_data  : Dict[str, pd.DataFrame] = {}
        self.test_data : Dict[str, pd.DataFrame] = {}
        
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
        import csv
        
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
        
        idx: int = torch.randint(len(data[folder].index) - 1, (1,)).item()
        # idx = random.randint(0, len(data[folder].index) - 2)
        
        I, id = self.get_random_image_from_folder_idx(folder, idx, list_type)
        
        return I, id, idx
      

    def get_random_image_from_folder_idx(self, folder: str, idx: int, list_type: ListType) -> Tuple[npt.NDArray[np.uint8], float]:
        """
        Get an image from a specific folder and index.
        
        For training: Creates a synthetic image by randomly interpolating between two adjacent images.
        For test/val: Returns the actual raw image without interpolation for consistent evaluation.

        Args:
            folder: Name of the folder containing the images
            idx: Starting index (for training, will use idx and idx+1 for interpolation)
            list_type: The type of list to get images from ('train', 'test', or 'val')

        Returns:
            Tuple of (image array, id)
        """
        I1, id1 = self.get_raw_image(folder, idx, list_type)
        
        # Only interpolate for training data
        if list_type == 'train':
            I2, id2 = self.get_raw_image(folder, idx+1, list_type)
            
            # Use U-shaped Beta distribution: Beta(0.5, 0.5)
            # This favors alpha near 0 and 1 (original images) while still allowing interpolation
            # More original images in training while keeping smooth transitions
            alpha: float = torch.distributions.Beta(0.5, 0.5).sample().item()
            I = cv.addWeighted(I1, alpha, I2, 1-alpha, 0)
            id: float = alpha*id1 + (1-alpha)*id2
        else:
            # Test/val: use actual raw images (no interpolation)
            I, id = I1, id1
        
        return I, id


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
    
    
    
