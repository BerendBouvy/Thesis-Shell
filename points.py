import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple


def grid_heatmap(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    cell_size: float,
    padding: float = 0.0,
    ax: Optional[plt.Axes] = None,
    cmap: str = "inferno",
    show: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Create a regular lat/lon grid, count points per cell, and plot a heatmap.

    Args:
        latitudes: Iterable of latitude values.
        longitudes: Iterable of longitude values.
        cell_size: Grid cell size in degrees (dx == dy).
        padding: Extra degrees added to every side of the bounding box.
        ax: Optional matplotlib axes; if None a new figure/axes is created.
        cmap: Colormap used for the heatmap.
        show: When True, call plt.show() after plotting.
        save_path: If provided, save the figure to this path.

    Returns:
        counts: 2D array with point counts per grid cell (lat x lon).
        (lat_edges, lon_edges): The bin edges defining the grid.
    """
    lat_arr = np.asarray(latitudes, dtype=float)
    lon_arr = np.asarray(longitudes, dtype=float)

    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    valid_mask = ~np.isnan(lat_arr) & ~np.isnan(lon_arr)
    lat_arr = lat_arr[valid_mask]
    lon_arr = lon_arr[valid_mask]

    if lat_arr.size == 0 or lon_arr.size == 0:
        raise ValueError("No valid latitude/longitude pairs provided")

    lat_min, lat_max = lat_arr.min() - padding, lat_arr.max() + padding
    lon_min, lon_max = lon_arr.min() - padding, lon_arr.max() + padding

    lat_bins = np.arange(lat_min, lat_max + cell_size, cell_size)
    lon_bins = np.arange(lon_min, lon_max + cell_size, cell_size)

    if lat_bins.size < 2:
        lat_bins = np.array([lat_min, lat_max])
    if lon_bins.size < 2:
        lon_bins = np.array([lon_min, lon_max])

    counts, lat_edges, lon_edges = np.histogram2d(lat_arr, lon_arr, bins=[lat_bins, lon_bins])

    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mesh = ax.pcolormesh(lon_edges, lat_edges, counts, cmap=cmap, shading="auto")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Point density heatmap (cell={cell_size}°)")
    plt.colorbar(mesh, ax=ax, label="Count")

    if save_path:
        (fig or ax.figure).savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    elif fig is not None:
        plt.close(fig)

    return counts, (lat_edges, lon_edges)
