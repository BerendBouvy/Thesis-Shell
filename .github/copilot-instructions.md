# Copilot Instructions for Thesis-Shell

## Project Overview
Thesis-Shell is a geospatial data processing pipeline for bathymetric/hydrographic survey data. It loads point cloud datasets from multiple surveys, performs coordinate transformations, analyzes spatial density, and generates raster grids and visualizations.

## Architecture & Data Flow

### Core Components

**1. `dataloader.py` - Data Pipeline**
- `dataLoader` class: Loads individual survey datasets from CSV/TXT files in `data/` directory
  - Reads metadata from `meta/metadata_with_density_flagged2.csv` (source of truth)
  - Auto-detects column names: `Lat (DMS)`, `Long (DMS)`, `Northing`, `Easting`, `Mean (m)`
  - Converts DMS coordinates to decimal degrees via `coordFunc.py`
  - Always converts to UTM Zone 31N (EPSG:32631) for consistency
  - Filters datasets: only includes records with `rejected == 0` AND `point_density(100x100m) > 20`
- File naming convention: `000574_XYZ_{LOCAL_CDI_ID}.txt`
- Output: Pickled list of `dataLoader` objects cached as `data_loaders_v2.pkl`

**2. `coordFunc.py` - Coordinate Utilities**
- Multi-datum support: WGS84 (4326), UTM Zone 31N (32631), UTM Zone 32N (32632)
- Key functions:
  - `dms_to_dd()`: Converts DMS strings (format: `52-30-45.123-N`) to decimal degrees
  - `convert_northing_easting()`: Transforms between coordinate systems using pyproj
  - `analyze_data_density()`: Calculates point density in 100×100m grid cells
  - `get_convex_hull()`: Computes convex hull in UTM space, returns area (sq km) and point density

**3. `aoi.py` - Area of Interest**
- `AreaOfInterest` class: Defines study region polygon from `AOI.txt`
- `get_raster()`: Generates density raster grid (1000×1000m default) across all dataloaders
  - Creates GeoTIFF output (`aoi_density.tif`) with proper geospatial metadata
  - Counts points per grid cell for visualization

**4. `points.py` - Visualization**
- `grid_heatmap()`: Creates lat/lon heatmaps with configurable cell size
- Handles NaN filtering, padding, and matplotlib/cartopy rendering
- Supports batch saving/plotting with Cartopy projections

### Data Flow
```
metadata_with_density_flagged2.csv
    ↓
dataLoader.load_data() → load_coordinates() → get_N31_coordinates()
    ↓
data_loaders_v2.pkl (pickled list)
    ↓
AreaOfInterest.get_raster() → aoi_density.tif (GeoTIFF raster)
    ↓
grid_heatmap() → PNG plots (plots/ directory)
```

## Developer Workflows

### Loading Data
```python
# Typical workflow (see test.py for examples)
import pickle
from dataloader import dataLoader

with open("data_loaders_v2.pkl", "rb") as f:
    loaders = pickle.load(f)

for loader in loaders:
    print(f"CDI ID: {loader.metadata['LOCAL_CDI_ID']}")
    print(f"Points: {len(loader)}")
    lats = loader.data['Lat'].to_list()
    lons = loader.data['Lon'].to_list()
```

### Regenerating Data Loaders
- Run `dataloader.create_data_loaders()` if metadata changes
- This loops through all records, applies filtering, and saves `data_loaders_v2.pkl`
- Progress: Uses `tqdm` for iteration feedback

### Creating Visualizations
- `create_plots()`: Generates individual dataset plots with bounding boxes
- `gantt_chart()`: Timeline visualization of collection periods
- `plot_number_of_points()`: Density heatmap across entire study region
- `hm()`: High-resolution point density (100m cells)

## Critical Patterns & Conventions

### Coordinate Systems
- **Storage**: Lat/Lon in WGS84 (EPSG:4326) + UTM Zone 31N as `Easting_N31`/`Northing_N31`
- **Default CRS**: `AreaOfInterest` uses `EPSG:32631` (UTM Zone 31N)
- **Visualization**: Cartopy with `ccrs.PlateCarree()` for plotting

### Metadata & Filtering
- `metadata_with_density_flagged2.csv`: Ground truth for datasets
- Critical fields: `CDI-record id`, `LOCAL_CDI_ID`, `Datum`, `rejected`, `point_density(100x100m)`
- Always check `rejected == 0` before processing
- Density threshold: `> 20 points per 100×100m cell`

### File Organization
- Input data: `data/*.txt` (raw survey files)
- Metadata: `meta/*.csv` (source of truth)
- Outputs: `plots/`, `*.tif` (raster grids)
- Cached objects: `*.pkl` (pickled dataLoader lists)

### Column Name Handling
- Function `check_delimiter()`: Auto-detects space vs tab delimiters
- Column names vary by dataset (handled in `load_data()`)
- Standard output columns: `Lat`, `Lon`, `Mean (m)`, `Easting_N31`, `Northing_N31`

## Common Extensions

**To add new analysis:**
1. Load `data_loaders_v2.pkl`
2. Iterate through loaders, access `.data` (pandas DataFrame) and `.metadata` dict
3. Use coordinate conversion functions from `coordFunc.py` as needed
4. For spatial analysis: use `.get_convex_hull()` or extract lat/lon for grid operations

**To add new visualization:**
- Use `points.grid_heatmap()` for lat/lon grids
- Use cartopy + matplotlib for map-based plots
- Save to `plots/` with version numbering pattern: `{name}_plot_V1.png` (auto-increments)

## Testing & Validation
- `test.py`: Contains utility functions for data validation
- `misc.ipynb`: Ad-hoc analysis notebook
- Check pickled cache exists before long workflows
- Verify `rejected` field before filtering conclusions

## Dependencies
- Core: `pandas`, `numpy`, `pyproj`, `geopandas`, `shapely`
- Viz: `matplotlib`, `cartopy`
- I/O: `rasterio` (GeoTIFF), `pickle`
- Progress: `tqdm`

## Notes for AI Agents
- Always check metadata `rejected` and `point_density` before assuming dataset validity
- Coordinate transformations are frequent—leverage `coordFunc.py` utilities
- Pickled dataLoader lists are large; cache results to avoid recomputation
- Plot versioning auto-increments; don't manually manage version numbers
