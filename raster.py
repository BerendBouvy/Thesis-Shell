import numpy as np
import rasterio
from rasterio.transform import from_origin
import rasterio.features as rfeatures

# Some rasterio builds expose `jurasterize` instead of `rasterize`.
rasterize = getattr(rfeatures, "rasterize", None)
if rasterize is None:
    rasterize = getattr(rfeatures, "jurasterize")

def gdf_to_raster(
    gdf,
    value_col="value",
    resolution=20,        # meters/pixel
    out_crs="EPSG:32631",
    fill=np.nan,
    all_touched=False,
    dtype="float32"
):
    # 1) Ensure CRS
    gdf = gdf.to_crs(out_crs)

    # 2) Build grid from bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    # top-left origin transform
    transform = from_origin(minx, maxy, resolution, resolution)

    # 3) Prepare (geometry, value) pairs
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[value_col]))

    # 4) Rasterize
    arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill if np.isfinite(fill) else 0,   # rasterize fill must be numeric
        all_touched=all_touched,
        dtype=dtype
    )

    # If you really want NaN background:
    if np.isnan(fill):
        arr = arr.astype("float32")
        arr[arr == 0] = np.nan

    return arr, transform, out_crs

def numpy_to_raster(array, bottom_left, dx=20, dy=20, crs="EPSG:32631", dtype="float32"):
    """transform numpy array to raster object"
    height, width = array.shape
    transform = from_origin(bottom_left[0], bottom_left[1] + height*dy, dx, dy)
    pass
    