#
# This is a class to process image from the disc and extract the nuclear layer from it
#
import cv2 as cv
import numpy as np
import numpy.typing as npt
import scipy.ndimage as nd
from skimage import measure
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Optional, Any, Literal
from pathlib import Path

from .functions import make_points_uniform, find_normals_inward, snap_to_pixels

import os

# Type alias for image types
ImageType = Literal['original', 'segmented', 'nuclear_layer', 'unrolled']


class CVImage:
    """
    Unified class for processing embryo images.
    
    Handles the complete pipeline from raw image to unrolled nuclear layer:
    1. Preprocessing (resizing, padding)
    2. Segmentation (embryo detection)
    3. Border detection and smoothing
    4. Border extension (inner/outer boundaries)
    5. Unrolling (perspective transformation to flatten nuclear layer)
    
    Optimized for performance with cached structuring elements and efficient array operations.
    All processing is done lazily - stages are computed only when needed.
    """
    
    # Class-level cache for structuring elements (shared across instances)
    _SE_CACHE: dict = {}
    
    def __init__(self, 
                 I: npt.NDArray[np.uint8], 
                 id: float,
                 size: Tuple[int, int] = (512, 512),
                 padding: int = 44,
                 plot_images: bool = False,
                 npoints: int = 100,
                 inward: int = 40,
                 outward: int = -24,
                 border_expansion: int = 0,
                 trunc_width: Optional[int] = None,
                 output_dir: Optional[str | Path] = None,
                 filename_stem: Optional[str] = None,
                 source_ppm: Optional[float] = None,
                 target_ppm: Optional[float] = 1.0) -> None:
        """
        Initialize the CVImage with an image and its ID.
        
        Args:
            I: Input grayscale image (will be copied to avoid modifying original)
            id: Image identifier/stage value
            size: Final target size after padding (default: (512, 512))
            padding: Border padding in pixels (default: 44)
            plot_images: Whether to display processing steps (default: False)
            npoints: Number of contour points (default: 100)
            inward: Distance to extend border inward in pixels (default: 40)
            outward: Distance to extend border outward in pixels (default: -24)
            border_expansion: Pixels to dilate segmentation outward (default: 5)
            trunc_width: Optional width to randomly crop the final unrolled image
            output_dir: Optional directory to save intermediate processing results
            filename_stem: Optional stem to use for filenames (e.g., 's1_c2_z1_t99')
                          If None, uses formatted id (e.g., '046303.00')
            source_ppm: Source pixels-per-micron from metadata (None = skip PPM scaling)
            target_ppm: Target pixels-per-micron for normalization (None or default: 1.0)
                       If None, PPM scaling is disabled even if source_ppm is available
        """
        # Store original image and id
        self.I_original = I.copy()  # Original input (never modified)
        
        # Store PPM parameters before applying scaling
        self.source_ppm = source_ppm
        self.target_ppm = target_ppm
        
        # Apply PPM scaling to original image if both source and target PPM are available
        # This is done once here rather than in each get_* method
        if source_ppm is not None and source_ppm > 0 and target_ppm is not None:
            scale_factor = target_ppm / source_ppm
            if abs(scale_factor - 1.0) >= 0.001:  # Only scale if factor differs significantly
                h, w = self.I_original.shape
                new_h = int(round(h * scale_factor))
                new_w = int(round(w * scale_factor))
                interp = cv.INTER_LINEAR if scale_factor > 1.0 else cv.INTER_AREA
                self.I_original = cv.resize(self.I_original, (new_w, new_h), interpolation=interp)
        
        # Convert to uint8 regardless of input bit depth
        # This ensures all downstream processing works with normalized uint8 images
        if self.I_original.dtype == np.uint16:
            # Scale from 16-bit (0-65535) to 8-bit (0-255)
            # Use bit shifting for exact conversion: >> 8 divides by 256
            self.I_original = (self.I_original >> 8).astype(np.uint8)
        elif self.I_original.dtype != np.uint8:
            # For other types, normalize to 0-255 range
            self.I_original = cv.normalize(self.I_original, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        self.id = id
        self.filename_stem = filename_stem if filename_stem is not None else f"{id:06.2f}"
        
        # Save arguments
        self.size = size
        self.padding = padding
        self.plot_images = plot_images
        self.npoints = npoints
        self.inward = inward
        self.outward = outward
        self.depthInImage = abs(self.outward) + abs(self.inward)
        self.border_expansion = border_expansion
        self.trunc_width = trunc_width
        self.output_dir = Path(output_dir) if output_dir is not None else None
        
        # Processed image properties
        self.dtype  = I.dtype
    
    @staticmethod
    def _get_structuring_element(shape: int, size: Tuple[int, int]) -> npt.NDArray:
        """
        Get a cached structuring element or create and cache it.
        
        Args:
            shape: Morphological shape (e.g., cv.MORPH_ELLIPSE)
            size: Size tuple (width, height)
            
        Returns:
            Structuring element array
        """
        key = (shape, size)
        if key not in CVImage._SE_CACHE:
            CVImage._SE_CACHE[key] = cv.getStructuringElement(shape, size)
        return CVImage._SE_CACHE[key]


    def _pad_and_resize(self, Img:npt.NDArray) -> npt.NDArray:
        """
        Preprocess the image by resizing and adding border padding.
        
        Resizes the image to (size - 2*padding) and then adds border padding,
        resulting in a final image of dimensions self.size.
        Updates image dimensions and dtype attributes.
        
        If output_dir is set, saves the preprocessed image automatically.
        """
        # Calculate resize dimensions: final_size - 2*padding
        resize_width  = self.size[0] - 2 * self.padding
        resize_height = self.size[1] - 2 * self.padding
        if resize_width <= 0 or resize_height <= 0:
            raise ValueError(
                f"Invalid dimensions: size {self.size} with padding {self.padding} "
                f"results in non-positive resize dimensions ({resize_width}, {resize_height})"
            )
        
        # Resize and add border in one pipeline
        Img = cv.resize(Img, (resize_width, resize_height), interpolation=cv.INTER_AREA)
        Img = cv.copyMakeBorder(
            Img, self.padding, self.padding, self.padding, self.padding, 
            cv.BORDER_REPLICATE
        )  # type: ignore
        
        # Update image properties
        self.dtype = Img.dtype
        
        # Save if output_dir is set
        if self.output_dir is not None:
            save_path = self.output_dir / f"{self.filename_stem}_01_preprocessed.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(save_path), Img)
        
        return Img
    
    
    def _detect_and_correct_illumination(self, Img: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], float, bool]:
        """
        Detect uneven illumination and apply flat-field correction if needed.
        
        Uses quadrant analysis to detect illumination gradients and applies
        the gold-standard flat-field correction formula from microscopy:
        Corrected = (Image - Dark) / (Flat - Dark) × Mean(Flat)
        
        Args:
            Img: Input grayscale image (uint8)
        
        Returns:
            Tuple of (corrected_image, illumination_cv, was_corrected)
                - corrected_image: Image with correction applied (if needed)
                - illumination_cv: Coefficient of variation for illumination
                - was_corrected: Boolean indicating if correction was applied
        """
        # Detect uneven illumination by analyzing intensity gradients
        # Split image into quadrants and compare their mean intensities
        h, w = Img.shape
        q1 = np.mean(Img[0:h//2, 0:w//2])
        q2 = np.mean(Img[0:h//2, w//2:w])
        q3 = np.mean(Img[h//2:h, 0:w//2])
        q4 = np.mean(Img[h//2:h, w//2:w])
        quadrant_std = np.std([q1, q2, q3, q4])
        quadrant_mean = np.mean([q1, q2, q3, q4])
        
        # Coefficient of variation: normalized measure of illumination uniformity
        # Values > 0.30 indicate significant uneven illumination
        # Conservative threshold to avoid edge artifacts from correction
        illumination_cv = quadrant_std / (quadrant_mean + 1e-6)
        
        was_corrected = False
        corrected_img = Img.copy()
        
        if illumination_cv > 0.30:
            # Apply flat-field correction using morphological opening
            # Use center-weighted blending to avoid edge artifacts
            # Use a very conservative kernel size (1/6 of image dimension)
            kernel_size = max(h, w) // 6
            # Ensure it's odd and reasonably sized
            kernel_size = max(11, min(kernel_size, 41))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Estimate background illumination using morphological opening
            # Opening removes small bright features, leaving only the background illumination pattern
            se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv.morphologyEx(Img, cv.MORPH_OPEN, se)
            
            # Apply Gaussian blur to smooth the background estimate
            blur_size = kernel_size // 2
            if blur_size % 2 == 0:
                blur_size += 1
            background = cv.GaussianBlur(background.astype(np.float32), (blur_size, blur_size), 0)
            
            # Dark frame: use 1st percentile as estimate
            dark_frame = np.percentile(Img, 1)
            
            # Apply flat-field correction formula
            bg_mean = np.mean(background)
            denominator = background - dark_frame
            denominator = np.where(denominator < 1, 1, denominator)
            
            corrected = ((Img.astype(np.float32) - dark_frame) / denominator) * bg_mean
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            # Create center-weighted blending mask
            # Full correction in center, no correction at edges, smooth transition
            # Use a 2D Gaussian centered on the image
            y_center, x_center = h // 2, w // 2
            y_grid, x_grid = np.ogrid[:h, :w]
            
            # Gaussian falloff - stronger in center, weaker at edges
            # sigma controls transition sharpness (larger = smoother transition)
            sigma_y = h / 3.0  # Covers about 3 sigma = 99.7% of image height
            sigma_x = w / 3.0
            
            gaussian_mask = np.exp(-((y_grid - y_center)**2 / (2 * sigma_y**2) + 
                                     (x_grid - x_center)**2 / (2 * sigma_x**2)))
            
            # Blend: center gets corrected image, edges get original
            corrected_img = (Img.astype(np.float32) * (1 - gaussian_mask) + 
                           corrected.astype(np.float32) * gaussian_mask)
            corrected_img = corrected_img.astype(np.uint8)
            was_corrected = True
        
        return corrected_img, illumination_cv, was_corrected
    
    
    def _segment_embryo_image(self, Img: npt.NDArray) -> npt.NDArray[np.uint8]:
        """
        Segment the embryo from the background using advanced image processing.
        
        Optimized for speed by performing segmentation at reduced resolution (256px),
        then upsampling the binary mask. This provides 4-16× speedup with minimal
        accuracy loss since embryo boundaries are large, smooth features.
        
        Pipeline:
        1. Normalize image intensity
        2. Gaussian Blur for smoothing
        2.5. Uneven illumination detection & flat-field correction
        3. Otsu's thresholding
        4. Morphological cleaning (binary)
        5. Flood fill for hole filling
        6. Extract largest connected component
        7. Border expansion
        
        Returns:
            Binary mask of segmented embryo (same size as input)
        """
        # ========================================================================
        # OPTIMIZATION: 256px segmentation with padding for edge context
        # ========================================================================
        orig_h, orig_w = Img.shape[:2]
        
        # Add padding before downsampling to help with edge detection
        # Replicates edge pixels to avoid introducing artifacts
        pad = 44
        Img_padded = cv.copyMakeBorder(Img, pad, pad, pad, pad, cv.BORDER_REPLICATE)
        
        # Target size for segmentation (256px)
        seg_size = 256
        padded_h, padded_w = Img_padded.shape[:2]
        scale = seg_size / max(padded_h, padded_w)
        new_h = int(padded_h * scale)
        new_w = int(padded_w * scale)
        
        # Downsample using INTER_AREA (best for downsampling)
        Img_small = cv.resize(Img_padded, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        # Calculate padding in downsampled space for later cropping
        pad_small = int(pad * scale)
        
        show_plot = self.plot_images or self.output_dir is not None
        if show_plot:
            plt.figure(figsize=(25, 10))
        
        # ========================================================================
        # STEP 1: Normalize image intensity to full range [0, 255]
        # ========================================================================
        Itmp = cv.normalize(
            src=Img_small, 
            dst=None, 
            alpha=0, 
            beta=255, 
            norm_type=cv.NORM_MINMAX, 
            dtype=cv.CV_8U
        )
        if show_plot:
            plt.subplot(2, 5, 1)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 1: Normalization')
            plt.axis('off')
        
        # ========================================================================
        # STEP 2: Gaussian Blur for Smoothing
        # Reduces high-frequency noise to improve segmentation
        # Scaled for 256px (~half of 51×51 for full resolution)
        # ========================================================================
        Itmp = cv.GaussianBlur(Itmp, (21, 21), 0)
        
        if show_plot:
            plt.subplot(2, 5, 2)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 2: Gaussian Blur')
            plt.axis('off')
        
        # ========================================================================
        # STEP 2.5: Uneven Illumination Detection & Flat-Field Correction
        # Detects and corrects for uneven lighting (common in microscopy)
        # Uses self-estimated flat field when reference image is unavailable
        # ========================================================================
        Itmp, illumination_cv, was_corrected = self._detect_and_correct_illumination(Itmp)
        
        if show_plot:
            plt.subplot(2, 5, 3)
            plt.imshow(Itmp, cmap='gray')
            cv_text = f'CV={illumination_cv:.3f}'
            correction_text = ' (Corrected)' if was_corrected else ' (No correction)'
            plt.title(f'Step 2.5: Illumination Check\n{cv_text}{correction_text}')
            plt.axis('off')
        
        # ========================================================================
        # STEP 3: Otsu's Thresholding
        # Global threshold - precise boundaries
        # Previous steps handle illumination variations
        # ========================================================================
        _, Itmp = cv.threshold(Itmp, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        Itmp = cv.bitwise_not(Itmp)
        
        if show_plot:
            plt.subplot(2, 5, 4)
            plt.imshow(Itmp, cmap='gray')
            plt.title("Step 3: Otsu Threshold")
            plt.axis('off')
        
        # ========================================================================
        # STEP 4: Morphological Cleaning (on BINARY image)
        # Opening removes small noise, Closing fills small gaps
        # ========================================================================
        # Opening: remove small white noise
        se_open = self._get_structuring_element(cv.MORPH_ELLIPSE, (3, 3))
        Itmp = cv.morphologyEx(Itmp, cv.MORPH_OPEN, se_open)
        
        # Closing: fill small holes in embryo
        se_close = self._get_structuring_element(cv.MORPH_ELLIPSE, (21, 21))
        Itmp = cv.morphologyEx(Itmp, cv.MORPH_CLOSE, se_close)
        
        if show_plot:
            plt.subplot(2, 5, 5)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 4: Binary Morphological Cleaning')
            plt.axis('off')
        
        # ========================================================================
        # STEP 5: Fill Embryo Interior
        # Large morphological closing to connect ring and fill interior
        # ========================================================================
        # Use very large kernel to bridge the nuclear ring and fill interior
        se_large = self._get_structuring_element(cv.MORPH_ELLIPSE, (21, 21))
        Itmp = cv.morphologyEx(Itmp, cv.MORPH_CLOSE, se_large)
        
        if show_plot:
            plt.subplot(2, 5, 6)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 5: Fill Interior')
            plt.axis('off')
        
        # ========================================================================
        # STEP 6: Hole Filling via Flood Fill
        # Fills any remaining holes by identifying background from corners
        # ========================================================================
        Itmp_filled = Itmp.copy()
        h, w = Itmp.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv.floodFill(Itmp_filled, mask, (0, 0), 255)
        Itmp_filled = cv.bitwise_not(Itmp_filled)
        Itmp = cv.bitwise_or(Itmp, Itmp_filled)
        
        if show_plot:
            plt.subplot(2, 5, 7)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 6: Hole Filling')
            plt.axis('off')
        
        # ========================================================================
        # STEP 7: Extract Largest Connected Component (embryo)
        # Removes any remaining small artifacts
        # ========================================================================
        labels = measure.label(Itmp, connectivity=2)
        unique, counts = np.unique(labels, return_counts=True)
        
        if len(unique) > 1:
            # Get largest component (excluding background label 0)
            max_label = unique[1:][np.argmax(counts[1:])]
            Itmp = np.where(labels == max_label, 255, 0).astype(np.uint8)
        
        if show_plot:
            plt.subplot(2, 5, 8)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 7: Largest Component')
            plt.axis('off')
        
        # ========================================================================
        # STEP 8: Border Expansion
        # Dilate to ensure border encompasses all nuclei
        # ========================================================================
        if self.border_expansion > 0:
            se_dilate = self._get_structuring_element(
                cv.MORPH_ELLIPSE, 
                (self.border_expansion * 2 + 1, self.border_expansion * 2 + 1)
            )
            Itmp = cv.dilate(Itmp, se_dilate)
        
        if show_plot:
            plt.subplot(2, 5, 9)
            plt.imshow(Itmp, cmap='gray')
            plt.title('Step 8: Final Segmentation')
            plt.axis('off')
        
        # ========================================================================
        # Remove padding before upsampling
        # Crop back to original (unpadded) aspect ratio
        # ========================================================================
        if pad_small > 0:
            Itmp = Itmp[pad_small:-pad_small, pad_small:-pad_small]
        
        # ========================================================================
        # Upsample binary mask back to original dimensions
        # Use INTER_NEAREST to preserve binary values (0 or 255)
        # ========================================================================
        Itmp = cv.resize(Itmp, (orig_w, orig_h), interpolation=cv.INTER_NEAREST)
        
        # ========================================================================
        # Save and Display
        # ========================================================================
        if show_plot:
            plt.tight_layout()
            
            if self.output_dir is not None:
                save_path = self.output_dir / f"{self.filename_stem}_02_segmentation.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            
            if self.plot_images:
                plt.show()
            else:
                plt.close()
        
        return Itmp

    
    def _border_finder(
        self,
        Img: npt.NDArray[np.uint8],
        Ilabel: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Find and smooth the embryo border contour.
        
        Extracts contours from the segmented image, identifies the largest,
        and applies multi-stage smoothing to create uniformly distributed
        boundary points.
        
        Smoothing procedure:
        1. Make points uniform along contour
        2. Apply 10% smoothing with uniform filter
        3. Re-distribute points uniformly
        4. Apply fine 3-point smoothing
        5. Final uniform distribution
        
        Args:
            Ilabel: Binary segmentation mask
        
        Returns:
            Tuple of (x, y) border coordinates
        
        If output_dir is set, saves a visualization of the detected border.
        """
        # Ensure correct data type for contour detection
        label_img = cv.convertScaleAbs(Ilabel)
        
        # Find all external contours
        contours, _ = cv.findContours(
            label_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )  # type: ignore
        
        if not contours:
            raise ValueError("No contours found in segmented image")
        
        # Find contour with maximum area using max with key (more efficient)
        max_contour = max(contours, key=cv.contourArea)
        
        # Extract x and y coordinates
        x = max_contour[:, 0, 0].astype(np.float64)
        y = max_contour[:, 0, 1].astype(np.float64)
        npoints_org = len(x)
        
        # Multi-stage smoothing and uniform distribution
        # Stage 1: Initial uniform distribution
        x, y, _ = make_points_uniform(x, y, n=npoints_org)
        
        # Stage 2: Coarse smoothing (10% of points)
        nsmooth = max(3, int(0.10 * npoints_org))
        x = nd.uniform_filter1d(x, size=nsmooth, mode='wrap')
        y = nd.uniform_filter1d(y, size=nsmooth, mode='wrap')
        
        # Stage 3: Re-distribute to target number of points
        x, y, _ = make_points_uniform(x, y, n=self.npoints)
        
        # Stage 4: Fine smoothing (3-point kernel)
        x = nd.uniform_filter1d(x, size=3, mode='wrap')
        y = nd.uniform_filter1d(y, size=3, mode='wrap')
        
        # Stage 5: Final uniform distribution
        x, y, _ = make_points_uniform(x, y, n=self.npoints)
        
        # Save if output_dir is set
        if self.output_dir is not None:
            save_path = self.output_dir / f"{self.filename_stem}_03_border.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create visualization: preprocessed image with border overlay
            vis_img = cv.cvtColor(Img, cv.COLOR_GRAY2BGR)
            # Draw the border points
            points = np.column_stack((x, y)).astype(np.int32)
            cv.polylines(vis_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv.imwrite(str(save_path), vis_img)
        
        return x, y
    
    
    def _extend_border(
        self,
        Img: npt.NDArray[np.uint8],
        x: npt.NDArray[np.float64], 
        y: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Extend the border inward and outward to define nuclear layer boundaries.
        
        Calculates unit normal vectors at each border point and extends them
        by specified distances to create the nuclear layer boundaries.
        
        The detected border (x, y) is approximately at the middle of the
        nuclear layer, which is located at the outer surface of the embryo.
        
        Args:
            x: X coordinates of border points
            y: Y coordinates of border points
        
        Returns:
            Tuple of (xext, yext, xint, yint) - outer and inner boundary coordinates
        
        If output_dir is set, saves a visualization with color coding:
            - Green: Original detected border (middle of nuclear layer)
            - Red: Outer edge of nuclear layer - top surface (xext, yext)
            - Blue: Inner edge of nuclear layer - bottom surface (xint, yint)
        """
        # Compute nuclear layer boundaries using normal vectors
        # Parameter mapping:
        #   self.outward (negative, e.g., -10) → extends outward → outer edge (top)
        #   self.inward  (positive, e.g., +22) → extends inward → inner edge (bottom)
        
        xext, yext = find_normals_inward(
            x=x, y=y, length=self.outward  # Negative → outward → outer edge (top)
        )
        xint, yint = find_normals_inward(
            x=x, y=y, length=self.inward  # Positive → inward → inner edge (bottom)
        )
        # Save if output_dir is set
        if self.output_dir is not None:
            save_path = self.output_dir / f"{self.filename_stem}_04_borders_extended.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create visualization: preprocessed image with both borders
            vis_img = cv.cvtColor(Img, cv.COLOR_GRAY2BGR)
            # Draw the original border (green)
            points = np.column_stack((x, y)).astype(np.int32)
            cv.polylines(vis_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            # Draw the outer border (red)
            points_ext = np.column_stack((xext, yext)).astype(np.int32)
            cv.polylines(vis_img, [points_ext], isClosed=True, color=(0, 0, 255), thickness=2)
            # Draw the inner border (blue)
            points_int = np.column_stack((xint, yint)).astype(np.int32)
            cv.polylines(vis_img, [points_int], isClosed=True, color=(255, 0, 0), thickness=2)
            cv.imwrite(str(save_path), vis_img)
        
        return xext, yext, xint, yint
    
    
    def _unroll(
        self,
        Img: npt.NDArray[np.uint8],
        xext: npt.NDArray[np.float64],
        yext: npt.NDArray[np.float64],
        xint: npt.NDArray[np.float64],
        yint: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.uint8]:
        """
        Unroll the nuclear layer from curved to flat representation.
        
        Applies perspective transformations segment by segment to unwrap the
        nuclear layer between inner and outer borders.
        
        Args:
            Img: Padded and resized image to unroll
            xext: X coordinates of outer boundary
            yext: Y coordinates of outer boundary
            xint: X coordinates of inner boundary
            yint: Y coordinates of inner boundary
        
        Returns:
            Unrolled image with shape (depth, width, channels)
        
        If output_dir is set, saves the unrolled image automatically.
        """
        x_nuc = xext
        y_nuc = yext
        depthInImage = self.depthInImage
        
        # Apply top-hat morphological transformation
        I = Img.copy()
        se_tophat = self._get_structuring_element(cv.MORPH_ELLIPSE, (20, 20))
        I = cv.morphologyEx(I, cv.MORPH_TOPHAT, se_tophat)
        
        # Convert to 3D array (add channel dimension)
        I = I[:, :, np.newaxis]
        m, n, o = I.shape
        # Snap border points to valid pixel coordinates
        xext_snapped  = snap_to_pixels(xext, n - 1)
        yext_snapped  = snap_to_pixels(yext, m - 1)
        xint_snapped  = snap_to_pixels(xint, n - 1)
        yint_snapped  = snap_to_pixels(yint, m - 1)
        x_nuc_snapped = snap_to_pixels(x_nuc, n - 1)
        y_nuc_snapped = snap_to_pixels(y_nuc, m - 1)

        # Create transformation point matrices
        X_t = np.column_stack((xext_snapped[:-1], xext_snapped[1:], xint_snapped[:-1], xint_snapped[1:]))
        Y_t = np.column_stack((yext_snapped[:-1], yext_snapped[1:], yint_snapped[:-1], yint_snapped[1:]))

        # Calculate bounding boxes for each segment
        xL = np.floor(np.min(X_t, axis=1)).astype(int)
        yL = np.floor(np.min(Y_t, axis=1)).astype(int)
        xU = np.ceil(np.max(X_t, axis=1)).astype(int)
        yU = np.ceil(np.max(Y_t, axis=1)).astype(int)

        # Translate points to local coordinates
        X_t1 = X_t - xL[:, None]
        Y_t1 = Y_t - yL[:, None]

        # Calculate segment widths
        dx = np.diff(x_nuc_snapped, axis=0)
        dy = np.diff(y_nuc_snapped, axis=0)
        ds = np.round(np.sqrt(dx**2 + dy**2)).astype(int)
        w = np.sum(ds)
        
        # Initialize output array
        U = np.zeros((depthInImage, w, o), dtype=np.uint8)
        ustart = 0
        ns = len(xext_snapped) - 1

        # Process each segment with perspective transformation
        for i in range(ns):
            # Extract image segment
            I1 = I[yL[i]:yU[i], xL[i]:xU[i], :]
            
            # Define transformation points
            inpts = np.float32(np.column_stack((X_t1[i, :], Y_t1[i, :])))
            op = np.array([[0, 0], [ds[i], 0], [0, depthInImage], [ds[i], depthInImage]])
            outpts = np.float32(op)

            # Compute and apply perspective transform
            M = cv.getPerspectiveTransform(inpts, outpts)  # type: ignore
            size = (int(op[1, 0]), int(op[2, 1]))
            It = cv.warpPerspective(I1, M, size, flags=cv.INTER_LINEAR)  # type: ignore
            
            # Ensure correct dimensionality (warpPerspective may return 2D for single channel)
            if It.ndim == 2:
                It = It[:, :, np.newaxis]

            # Assign transformed segment to output
            ufinish = ustart + ds[i]
            U[:, ustart:ufinish, :] = It
            ustart = ufinish
        
        # Save if output_dir is set
        if self.output_dir is not None:
            save_path = self.output_dir / f"{self.filename_stem}_05_unrolled.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the unrolled image (squeeze channel dimension if single channel)
            img_to_save = U.squeeze() if U.shape[2] == 1 else U
            cv.imwrite(str(save_path), img_to_save)
        
        return U
    

    
    def get_original_image(self) -> npt.NDArray[np.uint8]:
        """
        Get the original image resized to target dimensions but not padded.
        
        Returns:
            Resized original image (img_height, img_width), normalized to uint8
        """
        img = cv.resize(
            self.I_original, 
            (self.size[0], self.size[1]), 
            interpolation=cv.INTER_AREA
        )
        return img
    
    
    def get_segmented_image(self) -> npt.NDArray[np.uint8]:
        """
        Get the embryo image with background removed (masked by segmentation).
        
        Returns:
            Grayscale image of embryo with background set to black (0), normalized to uint8
        """
        # Pad and resize the original image
        img = self._pad_and_resize(self.I_original.copy())
        
        # Get the segmentation mask
        Ilabel = self._segment_embryo_image(img)
        
        # Apply mask to the padded image to get embryo pixels only
        result = cv.bitwise_and(img, img, mask=Ilabel)
        return result
    
    
    def get_nuclear_layer_image(self) -> npt.NDArray[np.uint8]:
        """
        Get a binary mask of just the nuclear layer region (not unrolled).
        
        Creates a mask showing the region between the inner and outer boundaries
        of the nuclear layer on the original padded image.
        
        Returns:
            Binary mask of nuclear layer region, normalized to uint8
        """
        # Pad and resize the original image
        img = self._pad_and_resize(self.I_original.copy())
        
        # Get the segmented image and border
        Ilabel = self._segment_embryo_image(img)
        x, y = self._border_finder(img, Ilabel)
        xext, yext, xint, yint = self._extend_border(img, x, y)
        
        # Create empty mask with dimensions matching the padded image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Create polygon points for inner and outer boundary
        outer_points = np.column_stack((xext, yext)).astype(np.int32)
        inner_points = np.column_stack((xint, yint)).astype(np.int32)
        
        # Fill outer boundary and subtract inner boundary (create ring)
        cv.fillPoly(mask, [outer_points], 255)
        cv.fillPoly(mask, [inner_points], 0)
        
        result = cv.bitwise_and(img, img, mask=mask)
        return result
    
    
    def get_unrolled_image(self, trunc_width: Optional[int] = None) -> npt.NDArray[np.uint8]:
        """
        Get the unrolled nuclear layer image.
        
        Args:
            trunc_width: Optional width for random cropping. If None, uses 
                        self.trunc_width from initialization. If that's also None,
                        returns full width.
        
        Returns:
            Unrolled nuclear layer image with optional width truncation, normalized to uint8
        """
        # Pad and resize the original image
        img = self._pad_and_resize(self.I_original.copy())
        
        # Get all the required data
        Ilabel = self._segment_embryo_image(img)
        x, y = self._border_finder(img, Ilabel)
        xext, yext, xint, yint = self._extend_border(img, x, y)
        Iunrolled = self._unroll(img, xext, yext, xint, yint)
        
        # Determine truncation width
        if trunc_width is None:
            trunc_width = self.trunc_width
        
        # Apply optional truncation
        if trunc_width is not None:
            max_start = Iunrolled.shape[1] - trunc_width
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            result = Iunrolled[:, start:start + trunc_width, :]
        else:
            result = Iunrolled
        return result
    
    
    
    def get_image(
        self, 
        image_type: ImageType = 'unrolled',
        trunc_width: Optional[int] = None
    ) -> npt.NDArray[np.uint8]:
        """
        Get image based on the specified type.
        
        Args:
            image_type: Type of image to return:
                - 'original': Original input image resized but not padded
                - 'segmented': Binary segmentation mask of entire embryo
                - 'nuclear_layer': Binary mask of nuclear layer region (not unrolled)
                - 'unrolled': Unrolled nuclear layer with optional width truncation
            trunc_width: Optional width for random cropping (only for 'unrolled' type).
                        If None, uses self.trunc_width from initialization.
        
        Returns:
            Image array of the requested type
            
        Raises:
            ValueError: If image_type is invalid or processing failed
        """
        if image_type == 'original':
            return self.get_original_image()
        
        elif image_type == 'segmented':
            return self.get_segmented_image()
        
        elif image_type == 'nuclear_layer':
            return self.get_nuclear_layer_image()
        
        elif image_type == 'unrolled':
            return self.get_unrolled_image(trunc_width=trunc_width)
        
        else:
            raise ValueError(
                f"Invalid image_type '{image_type}'. Must be one of: "
                f"'original', 'segmented', 'nuclear_layer', 'unrolled'"
            )
    



if __name__ == "__main__":
    print("Hello from CVImage")
