import numpy as np
from matplotlib import pyplot as plt
import contextily as ctx
from shapely.geometry import box
import geopandas as gpd

def calculate_zoom(w, s, e, n):
    """Automatically choose a zoom level given a desired number of tiles.

    .. note:: all values are interpreted as latitude / longitutde.

    Parameters
    ----------
    w : float
        The western bbox edge.
    s : float
        The southern bbox edge.
    e : float
        The eastern bbox edge.
    n : float
        The northern bbox edge.

    Returns
    -------
    zoom : int
        The zoom level to use in order to download this number of tiles.
    """
    # Calculate bounds of the bbox
    lon_range = np.sort([e, w])[::-1]
    lat_range = np.sort([s, n])[::-1]

    lon_length = np.subtract(*lon_range)
    lat_length = np.subtract(*lat_range)

    # Calculate the zoom
    zoom_lon = np.log2(360 * 2.0 / lon_length)
    zoom_lat = np.log2(360 * 2.0 / lat_length)
    zoom = np.min([zoom_lon, zoom_lat])
    return zoom

def zoom_from_viewport(viewport, crs=3857):
    zoom = calculate_zoom(*gpd.GeoDataFrame(geometry=[box(viewport["xmin"], viewport["ymin"], viewport["xmax"], viewport["ymax"])], crs=crs).to_crs(4326).unary_union.bounds)
    return zoom

def prepare_map_plot(figsize, dpi):
    # Create figure
    fig, ax = plt.subplots(1,1,figsize=figsize, dpi=dpi)
    return fig, ax

def viewport_from_gdf(gdf):
    bbox = gdf.to_crs(epsg=3857).geometry.unary_union.bounds
    vp = {"xmin":bbox[0], "xmax":bbox[2], "ymin":bbox[1], "ymax":bbox[3]}
    return vp

def gen_viewport_gdf(xmin, xmax, ymin, ymax, crs=3857):
    return gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=crs)

def plot_gpds_on_basemap(ax, xmin, xmax, ymin, ymax, basemap_source, zoom, gpds, style_kws):

    # Force figure size
    ax.set_adjustable("datalim")

    # Plot invisible viewport box
    gen_viewport_gdf(xmin, xmax, ymin, ymax, crs=3857).plot(ax=ax, edgecolor="None", color="None")

    # Extract viewport axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot payload data
    for i, d in enumerate(gpds): 
        d.to_crs(epsg=3857).plot(ax=ax, **style_kws[i])

    # Set limits
    ax.set(
        xlim = xlim,
        ylim = ylim,
        yticks=[],
        xticks=[]
    )

    # Add basemap
    ctx.add_basemap(ax=ax,  
                source=basemap_source, 
                attribution=False,
                zoom = zoom)

    return ax