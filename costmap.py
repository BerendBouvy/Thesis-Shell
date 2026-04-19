
import aoi
import numpy as np


def nansum(a, axis=None):
    """Like np.nansum, but returns NaN where all elements along the axis are NaN."""
    a = np.asarray(a, dtype=float)
    all_nan = np.all(np.isnan(a), axis=axis)
    result = np.nansum(a, axis=axis)
    result[all_nan] = np.nan
    return result


class CostMap:
    def __init__(self, dx, dy, default_cost=1.0, start_utm=(560123, 5931050), end_utm=(629112, 5942337), csr="EPSG:32631"):
        self.nx = 80e3
        self.ny = 35e3
        self.dx = dx
        self.dy = dy
        self.costs = np.full((int(self.ny//self.dy), int(self.nx//self.dx)), default_cost, dtype=float)
        self.cm_height, self.cm_width = self.costs.shape
        self.start_utm = start_utm
        self.end_utm = end_utm
        self.csr = csr
        self.bl = 555652, 5910512
        self.load_routes()
        self.ur = self.bl[0] + 80e3, self.bl[1] + 35e3
        self.cell_size = 5e3  # 5 km cells for slicing
        
        
    def read_AOI(self, file="AOI.txt"):
        """Read AOI.txt and return a list of (x, y) tuples."""
        points = []
        with open(file, 'r') as f:
            for line in f:
                line = line.replace(',', '.')
                x_str, y_str = line.strip().split('\t')
                points.append((float(x_str), float(y_str)))
                
        self.ll = points[0]
        self.lr = points[1]
        self.ur = points[2]
        self.ul = points[3]
        self.corners = [self.ll, self.lr, self.ur, self.ul]        
        return points
    
    def set_cost(self, x_idx, y_idx, cost, x_idx_end=None, y_idx_end=None):
        """Set the cost for a specific cell in the cost map."""
        if x_idx is None and y_idx is None:
            if self.costs.shape == cost.shape:
                self.costs = cost
            else:                
                raise ValueError("Cost array shape does not match cost map shape")
            return
        
        if x_idx_end is None:
            x_idx_end = x_idx + 1
        if y_idx_end is None:
            y_idx_end = y_idx + 1

        if 0 <= x_idx < self.cm_width and 0 <= y_idx < self.cm_height and 0 < x_idx_end <= self.cm_width and 0 < y_idx_end <= self.cm_height:
            self.costs[y_idx:y_idx_end, x_idx:x_idx_end] = cost
        else:
            raise IndexError("Cell index out of bounds")
        
    def add_cost(self, x_idx, y_idx, cost, x_idx_end=None, y_idx_end=None):
        """Add to the cost for a specific cell in the cost map."""
        if x_idx is None and y_idx is None:
            if self.costs.shape == cost.shape:
                self.costs = nansum([self.costs, cost], axis=0)
            else:                
                raise ValueError("Cost array shape does not match cost map shape")
            return
        if x_idx_end is None:
            x_idx_end = x_idx + 1
        if y_idx_end is None:
            y_idx_end = y_idx + 1

        if 0 <= x_idx < self.cm_width and 0 <= y_idx < self.cm_height and 0 < x_idx_end <= self.cm_width and 0 < y_idx_end <= self.cm_height:
            self.costs[y_idx:y_idx_end, x_idx:x_idx_end] = nansum([self.costs[y_idx:y_idx_end, x_idx:x_idx_end], cost], axis=0)
        else:
            raise IndexError("Cell index out of bounds")
        
    def multiply_cost(self, x_idx, y_idx, factor, x_idx_end=None, y_idx_end=None):
        """Multiply the cost for a specific cell in the cost map by a factor."""
        if x_idx_end is None:
            x_idx_end = x_idx + 1
        if y_idx_end is None:
            y_idx_end = y_idx + 1

        if 0 <= x_idx < self.cm_width and 0 <= y_idx < self.cm_height and 0 < x_idx_end <= self.cm_width and 0 < y_idx_end <= self.cm_height:
            self.costs[y_idx:y_idx_end, x_idx:x_idx_end] *= factor
        else:
            raise IndexError("Cell index out of bounds")
        
    def get_cost(self, x_idx, y_idx):
        """Get the cost for a specific cell in the cost map."""
        if 0 <= x_idx < self.cm_width and 0 <= y_idx < self.cm_height:
            return self.costs[y_idx, x_idx]
        else:
            raise IndexError("Cell index out of bounds")
        
    def set_nans(self, nanmap):
        """Set cost to NaN where nanmap is NaN."""
        if self.costs.shape == nanmap.costs.shape:
            self.costs[np.isnan(nanmap.costs)] = np.nan
        else:
            raise ValueError("Nan map shape does not match cost map shape")
        
    def plot_cost_map(self, raster=None, ax=None, cmap='Blues', show=True,
                      save_path=None, show_routes=False):
        """Visualize the cost map using a heatmap.

        Parameters
        ----------
        show_routes : bool
            Overlay the proposed routes stored in self.routes on top of the
            cost map.  Each route is drawn in a distinct colour with a legend
            entry.  Requires load_routes() to have been called first.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        data = raster if raster is not None else self.costs
        cax = ax.imshow(data, origin='lower', cmap=cmap)
        ax.set_title("Cost Map")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")
        plt.colorbar(cax, ax=ax, label='Cost')

        if show_routes:
            routes = self.routes
            route_colors = ["red", "cyan", "magenta", "yellow", "orange",
                            "lime", "white", "deepskyblue"]
            legend_handles = []
            for i, route in enumerate(routes):
                color = route_colors[i % len(route_colors)]
                # Build an RGBA overlay: transparent everywhere except route pixels
                overlay = np.zeros((*route.shape, 4), dtype=float)
                overlay[route, :] = plt.matplotlib.colors.to_rgba(color, alpha=0.9)
                ax.imshow(overlay, origin='lower')
                route_cost = float(self.costs[route].sum())
                legend_handles.append(
                    Patch(color=color, label=f"Route {i + 1}  (cost: {route_cost:.1f})")
                )
            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost map saved to {save_path}")

        if show:
            plt.show()
            
    def get_raster(self):
        """get a raster that corresponds to the utm 31n coordinates of the AOI"""
        pass
    
    def slice_cost_map(self, index):
        """Return a slice of the cost map for a given index
        corresponding to one of the 112 cells in the analysis grid.
        the cells are 5x5 km, and the aoi is 80x35 km, so the grid is 16x7 cells.
        bottom left cell is index 0, bottom right cell is index 15, top left cell is index 96, top right cell is index 111.
        """
        size = self.costs.shape[0]//7
        row = index // 16
        col = index % 16
        x_start = int(col * size)
        x_end = x_start + int(size)
        y_start = int(row * size)
        y_end = y_start + int(size)
        return x_start, x_end, y_start, y_end
    
    def get_idx_from_coordinates(self, x, y):
        """Convert UTM coordinates to cost map indices."""
        if not (self.bl[0] <= x <= self.ur[0]) or not (self.bl[1] <= y <= self.ur[1]):
            raise ValueError("Coordinates out of bounds")
        
        x_idx = int((x - self.bl[0]) / self.dx)
        y_idx = int((y - self.bl[1]) / self.dy)
        
        return x_idx, y_idx

    def block_n2000(self, array=None, path="shapes/n200.shp"):
        """
        Set all cost map cells that fall inside any N2000 polygon to -1 (no-go).

        Reads the shapefile at `path`, reprojects to the cost map CRS, then
        rasterizes each polygon onto self.costs.
        """
        import geopandas as gpd
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        
        if array is not None:
            if array.shape != self.costs.shape:
                raise ValueError("Input array shape does not match cost map shape")
            cost = array
        else:
            cost = self.costs

        gdf = gpd.read_file(path).to_crs(self.csr)
        geometries = [geom for geom in gdf.geometry if geom is not None]

        if not geometries:
            return

        affine = from_origin(
            west=self.bl[0],
            north=self.bl[1] + self.ny,
            xsize=self.dx,
            ysize=self.dy,
        )

        mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=cost.shape,
            transform=affine,
            fill=0,
            dtype=np.uint8,
        )

        # rasterize fills row 0 = north, but self.costs uses row 0 = south
        # (plotted with origin='lower'), so flip vertically before applying.
        cost[np.flipud(mask) == 1] = -1
        return cost
        
    def load_routes(self, file_path=["shapes/line1.shp", "shapes/line2.shp"]):
        """
        Load one or more route shapefiles and store a skeletonized raster
        representation of each in self.routes (list of bool arrays).

        Each route is rasterized to the same grid as self.costs, then
        skeletonized so that diagonal segments are represented as single
        diagonal steps rather than thick L-shaped staircases:

            0 0 1        0 0 1
            0 1 1  -->   0 1 0
            1 1 0        1 0 0
        """
        import geopandas as gpd
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        from skimage.morphology import skeletonize

        if isinstance(file_path, str):
            file_path = [file_path]

        affine = from_origin(
            west=self.bl[0],
            north=self.bl[1] + self.ny,
            xsize=self.dx,
            ysize=self.dy,
        )

        self.routes = []
        for path in file_path:
            gdf = gpd.read_file(path).to_crs(self.csr)
            geometries = [geom for geom in gdf.geometry if geom is not None]

            if not geometries:
                self.routes.append(np.zeros(self.costs.shape, dtype=bool))
                continue

            raster = rasterize(
                [(geom, 1) for geom in geometries],
                out_shape=self.costs.shape,
                transform=affine,
                fill=0,
                dtype=np.uint8,
            )

            # row 0 = north in rasterio, row 0 = south in self.costs
            raster = np.flipud(raster)

            # Thin to single-pixel-wide skeleton so diagonal steps are clean
            skeleton = skeletonize(raster.astype(bool))
            self.routes.append(skeleton)

        return self.routes
    
    def fill_nans_nn(self):
        """Fill NaN values in the cost map using nearest neighbor interpolation."""
        nan_mask = np.isnan(self.costs)
        if not nan_mask.any():
            return

        rows, cols = np.indices(self.costs.shape)
        valid = ~nan_mask
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([rows[valid], cols[valid]]))
        _, idx = tree.query(np.column_stack([rows[nan_mask], cols[nan_mask]]))
        valid_values = self.costs[valid]
        self.costs[nan_mask] = valid_values[idx]
        
    def fill_nans_high_cost(self, high_cost=None):
        """Fill NaN values in the cost map with a specified high cost."""
        if high_cost is None:
            high_cost = np.nanmax(self.costs)
        self.costs[np.isnan(self.costs)] = high_cost



if __name__ == "__main__":
    cm = CostMap(dx=100, dy=100)
    print("Cost map initialized with shape:", cm.costs.shape)
    for i in range(1, 113):
        slice = cm.slice_cost_map(i)
        cm.set_cost(slice[0], slice[2], cost=i, x_idx_end=slice[1], y_idx_end=slice[3])
    cm.plot_cost_map()
