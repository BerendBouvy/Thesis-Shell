import geopandas as gpd
import pyproj as pp
from shapely.geometry import Polygon


class AreaOfInterest:
    def __init__(self, file_path, crs="EPSG:4326"):
        self.file_path = file_path
        self.crs = crs
        self.aoi_gdf = self.load_aoi()
        
    def load_aoi(self):
        gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs="EPSG:4326")
        aoi_gdf = aoi_gdf.to_crs(self.crs)
        return aoi_gdf
    
    def get_aoi_polygon(self):
        if self.aoi_gdf.geometry.type.iloc[0] == 'Polygon':
            return self.aoi_gdf.geometry.iloc[0]
        elif self.aoi_gdf.geometry.type.iloc[0] == 'MultiPolygon':
            return self.aoi_gdf.geometry.iloc[0].geoms[0]
        else:
            raise ValueError("AOI geometry must be a Polygon or MultiPolygon.")
    
    def plot_aoi(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        self.aoi_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
        ax.set_title("Area of Interest")
        return ax
    
    def contains_point(self, latitude, longitude):
        point = gpd.GeoSeries([sg.Point(longitude, latitude)], crs="EPSG:4326")
        aoi_polygon = self.get_aoi_polygon()
        return point.within(aoi_polygon).iloc[0]
    
    def get_bounds(self):
        aoi_polygon = self.get_aoi_polygon()
        return aoi_polygon.bounds  # returns (minx, miny, maxx, maxy)
    
def filter_points_within_aoi(latitudes, longitudes, aoi: AreaOfInterest):
    aoi_polygon = aoi.get_aoi_polygon()
    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(longitudes, latitudes), crs="EPSG:4326")
    within_aoi = points_gdf.within(aoi_polygon)
    filtered_points = points_gdf[within_aoi]
    return filtered_points.geometry.y.tolist(), filtered_points.geometry.x.tolist()



if __name__ == "__main__":
    # Example usage
    aoi = AreaOfInterest("AOI.txt")
    aoi.plot_aoi()