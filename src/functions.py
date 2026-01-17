import os
import numpy as np
import numpy.typing as npt
from typing import Tuple



def make_points_uniform(
    x: npt.NDArray[np.floating], 
    y: npt.NDArray[np.floating], 
    n: int
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Redistribute contour points uniformly along arc length.
    
    Interpolates contour points to create evenly-spaced points along the
    contour perimeter. Ensures n points (adjusted to even if needed).
    Removes consecutive duplicate points deterministically.
    
    Args:
        x: x-coordinates of the contour points
        y: y-coordinates of the contour points
        n: Target number of points for the uniform contour
    
    Returns:
        Tuple of (x_uniform, y_uniform, normalized_arclength)
    """
    # Ensure even number of points
    n = n + 1 if n % 2 else n
    
    # Remove consecutive duplicate points deterministically
    points = np.column_stack((x, y))
    if len(points) > 1:
        # Calculate differences between consecutive points
        diffs = np.diff(points, axis=0)
        # Keep points where either x or y changed
        mask = np.any(diffs != 0, axis=1)
        # Always keep the first point, then apply mask to rest
        keep_indices = np.concatenate(([True], mask))
        points = points[keep_indices]
        x = points[:, 0]
        y = points[:, 1]
    
    # Close the contour by appending the first point
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    
    # Compute cumulative arc length of closed contour
    segment_lengths = np.hypot(np.diff(x_closed), np.diff(y_closed))
    arclen = np.concatenate(([0], np.cumsum(segment_lengths)))
    
    # Create uniform sampling along arc length
    perimeter = arclen[-1]
    s = np.linspace(0, perimeter, n + 1)
    
    # Interpolate to get uniformly spaced points
    x2 = np.interp(s, arclen, x_closed)
    y2 = np.interp(s, arclen, y_closed)
    x2[-1] = x2[0]  # Ensure exact closure
    y2[-1] = y2[0]
    
    # Compute normalized arc length for new contour
    new_lengths = np.hypot(np.diff(x2), np.diff(y2))
    arclen2 = 2 * np.concatenate(([0], np.cumsum(new_lengths))) / np.sum(new_lengths) - 1
    
    return x2, y2, arclen2



def find_normals_inward(
    x: npt.NDArray[np.floating], 
    y: npt.NDArray[np.floating], 
    length: float = 5
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Calculate points offset from contour along normal vectors.

    Computes normal vectors at each contour point and extends them by the
    specified length. Positive length extends inward, negative extends outward.

    Args:
        x: x-coordinates of the contour points
        y: y-coordinates of the contour points
        length: Distance to extend along normals (default: 5 pixels)
                Positive values extend inward (toward center, smaller circle)
                Negative values extend outward (away from center, larger circle)

    Returns:
        Tuple of (x_extended, y_extended) coordinates
    """
    # Compute tangent vectors using central differences
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)
    
    # Compute normalized normal vectors (perpendicular to tangent)
    norm = np.hypot(dx, dy)
    nx = np.where(norm > 0, -dy / norm, 0)
    ny = np.where(norm > 0, dx / norm, 0)
    
    # Extend points along normals (negate length to make positive = inward)
    # The normal vectors computed above point outward, so we negate to make
    # positive length move inward (toward center) as the function name suggests
    x2 = x + (-length) * nx
    y2 = y + (-length) * ny
    x2[-1] = x2[0]
    y2[-1] = y2[0]

    return x2, y2



def get_contours(
    x: npt.NDArray[np.floating], 
    y: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Convert x, y coordinates to OpenCV contour format.

    Creates a 3D array compatible with OpenCV's drawContours function.

    Args:
        x: x-coordinates of the contour points
        y: y-coordinates of the contour points

    Returns:
        Contour array with shape (n, 1, 2) for OpenCV compatibility
    """
    
    # make new contour & add an extra dimension to match the original shape of max_contour
    contour = np.stack((x, y), axis=-1)
    contour = contour[:, np.newaxis, :]
    
    return contour



def snap_to_pixels(
    coords: npt.NDArray[np.floating], 
    max_val: int
) -> npt.NDArray[np.floating]:
    """
    Snap floating-point coordinates to integer pixel indices.
    
    Rounds coordinates and clips them to valid pixel range [0, max_val].
    
    Args:
        coords: Floating-point coordinates to snap
        max_val: Maximum valid pixel index (typically image width or height - 1)
    
    Returns:
        Clipped and rounded coordinates as floating-point array
    """
    return np.clip(np.round(coords), 0, max_val)
