import matplotlib as mpl
import matplotlib.transforms as mpltransforms

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
               
# COLORS
def toHex(rgb_tuple):
    return '#%02x%02x%02x' % rgb_tuple

# FONT
def setFont(font_path):
    
    fontmng = mpl.font_manager

    fontmng.fontManager.addfont(font_path)
    prop = fontmng.FontProperties(fname=font_path)

    plt.rcParams.update({
        "font.family":prop.get_name()
    })

def setFontSize(axes, size):

    for ax in axes:
        legend_texts = ax.get_legend().get_texts() if ax.get_legend() is not None else []
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()+legend_texts):
            item.set_fontsize(size)

# RENDERING

def setLatexRendering(on):
    if on: 
        plt.rcParams.update({
            "text.usetex": True
        })
    if not on:
        plt.rcParams.update({
            "text.usetex": False
        })

## LEGEND HANDLES

def moveLegend(from_ax, to_ax, **legend_props):
    handles, labels = from_ax.get_legend_handles_labels()
    to_ax.legend(handles=handles.copy(), labels=labels.copy(), **legend_props)
    from_ax.get_legend().remove()

def removeAxis(ax):
    ax.axis('off')

# Generate a round legend handle
def genRoundLegendHandle(color, markersize, **kwargs):
    return Line2D([0], [0], marker='o', color='w', label='Circle', markerfacecolor=color, markersize=markersize, **kwargs)

# Generate a rect legend handle
def genRectLegendHandle(color, **kwargs):
    return Patch(color=color, **kwargs)

# Generate a line legend handle
def genLineLegendHandle(color, **kwargs):
    return Line2D([0], [1], color=color, label="Line", **kwargs)

# Generate a marker handle
def genMarkerHandle(marker, **kwargs):
    return Line2D([0], [0], marker=marker, **kwargs)

# list of valid arguments vor legend_mode
valid_legend_modes = ["above", "below", "left", "right"]

## Scatter plots

# import custom marker from SVG file
def load_svg_marker(file):
    path, attributes = svg2paths(file)
    mk = parse_path(attributes[0]['d'])
    mk.vertices -= mk.vertices.mean(axis=0)
    mk = mk.transformed(mpltransforms.Affine2D().rotate_deg(180))
    mk = mk.transformed(mpltransforms.Affine2D().scale(-1,1))
    return mk