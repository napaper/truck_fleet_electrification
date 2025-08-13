"""
Style configuration module for truck fleet electrification analysis.

This module provides consistent styling configurations for plots and visualizations,
including TUM corporate colors, plot dimensions, and color palettes for various
analysis scenarios.

The module ensures adherence to TUM corporate design guidelines and provides
consistent visual representation across all analysis outputs.
"""

from matplotlib.colors import ListedColormap
import colorsys


# =============================================================================
# PLOT DIMENSIONS AND LAYOUT
# =============================================================================

# Base text width for plots (optimized for FTM Thesis format)
TEXT_WIDTH = 6.32

# Height ratios for different plot aspect ratios
HEIGHT_16_9 = TEXT_WIDTH * 9 / 16
HEIGHT_4_3 = TEXT_WIDTH * 3 / 4

# Grid line transparency settings
ALPHA_MAJOR_GRID = 0.8
ALPHA_MINOR_GRID = 0.4


# =============================================================================
# TUM CORPORATE COLOR DEFINITIONS
# =============================================================================

# Primary TUM corporate colors following official brand guidelines
TUM_COLORS = {
    'TUMBlack': '#000000',      # Primary black
    'TUMWhite': '#FFFFFF',      # Primary white
    'TUMBlue1': '#005293',      # Primary TUM blue
    'TUMOrange': '#E37222',     # Primary TUM orange
    'TUMBlue2': '#3070b3',      # Secondary blue
    'TUMGreen3': '#A2AD00',     # Light green
    'TUMBlue3': '#64A0C8',      # Light blue
    'TUMGreen2': '#515600',     # Dark green
    'TUMBlue4': '#98C6EA',      # Very light blue
    'TUMGray1': '#333333',      # Dark gray
    'TUMBlue5': '#B0D0F0',      # Very light blue
    'TUMGray2': '#7F7F7F',      # Medium gray
    'TUMBlue6': '#2A4D70',      # Dark blue
    'TUMGray3': '#CCCCCC',      # Light gray
    'TUMBlueDark': '#0B1340',   # Very dark blue
    'TUMIvory': '#DAD7CB',      # Ivory/cream
    'TUMGreen1': '#292B00',     # Very dark green
    'Purple': '#B3679B',        # Purple accent
    'LightPurple': '#020202',   # Light purple
    'DarkPurple': '#31081F',    # Dark purple
    'LightYellow': '#FFE548',   # Light yellow
    'Rust': '#885053'           # Rust accent
}

# Filtered color palette excluding white and very light colors for line plots
LINE_COLORS = {
    key: value for key, value in TUM_COLORS.items() 
    if value not in ['#FFFFFF', '#CCCCCC']
}


# =============================================================================
# SPECIALIZED COLOR PALETTES
# =============================================================================

# Color palette for freight forwarder identification (1-6)
FREIGHT_FORWARDER_PALETTE = {
    1: TUM_COLORS['TUMBlue1'],
    2: TUM_COLORS['TUMOrange'],
    3: TUM_COLORS['Purple'],
    4: TUM_COLORS['TUMGreen3'],
    5: TUM_COLORS['TUMBlue4'],
    6: TUM_COLORS['TUMBlue2'],
}

# Color palette for weekday representation and combined data
WEEKDAY_PALETTE = {
    0: TUM_COLORS['TUMBlue4'],      # Monday
    1: TUM_COLORS['TUMBlue3'],      # Tuesday
    2: TUM_COLORS['TUMBlue2'],      # Wednesday 
    3: TUM_COLORS['TUMBlue1'],      # Thursday
    4: TUM_COLORS['TUMGray3'],      # Friday
    5: TUM_COLORS['TUMGray2'],      # Saturday 
    6: TUM_COLORS['TUMGray1'],      # Sunday
    'Combined': TUM_COLORS['LightPurple']  # Average/Combined data
}

# Scenario names for different charging and battery configurations
SCENARIO_NAMES = {
    'default': 'Default',
    'low_home': '50kW Home',
    'high_home': '350kW Home',
    'destination': 'Destination Charging',
    'big_batt': 'Large Battery',
    'ultra_home': 'Inf. Home Charging',
    'backup1': 'Backup 1',
    'backup2': 'Backup 2',
}


# =============================================================================
# COLOR UTILITY FUNCTIONS
# =============================================================================

def calculate_brightness(hex_color):
    """
    Calculate the brightness (luminance) of a hex color.
    
    Args:
        hex_color (str): Hex color code (e.g., '#FF0000')
        
    Returns:
        float: Brightness value between 0 (black) and 1 (white)
        
    Raises:
        ValueError: If hex_color is not a valid hex color string
    """
    try:
        # Convert hex to RGB values
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        
        # Convert RGB to HLS and extract luminance
        hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        return hls[1]  # Luminance value
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid hex color format: {hex_color}") from e


# =============================================================================
# PROCESSED COLOR LISTS
# =============================================================================

# Create sorted color list by brightness (descending)
COLOR_LIST = list(LINE_COLORS.values())
COLOR_LIST_SORTED = sorted(COLOR_LIST, key=calculate_brightness, reverse=True)

# Curated color list for specific plot types
CURATED_PLOT_COLORS = [
    COLOR_LIST_SORTED[10],  # Bright color
    COLOR_LIST_SORTED[2],   # Medium bright
    COLOR_LIST_SORTED[3],   # Medium
    COLOR_LIST_SORTED[4],   # Medium dark
    COLOR_LIST_SORTED[15],  # Dark
    COLOR_LIST_SORTED[18],  # Very dark
    COLOR_LIST_SORTED[19],  # Extremely dark
    COLOR_LIST_SORTED[1],   # Special case
    COLOR_LIST_SORTED[6]    # Special case
]

# Create matplotlib colormap from sorted TUM colors
TUM_COLORMAP = ListedColormap(COLOR_LIST_SORTED[:97])


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Maintain backward compatibility for existing code
textwidth = TEXT_WIDTH
h_169 = HEIGHT_16_9
h_43 = HEIGHT_4_3
alpha_major = ALPHA_MAJOR_GRID
alpha_minor = ALPHA_MINOR_GRID
colors = TUM_COLORS
line_colors = LINE_COLORS
palette_ff = FREIGHT_FORWARDER_PALETTE
palette_weekdays = WEEKDAY_PALETTE
scenario_names = SCENARIO_NAMES
brightness = calculate_brightness
color_list = COLOR_LIST
color_list_sorted = COLOR_LIST_SORTED
colors_plot = CURATED_PLOT_COLORS
cmap = TUM_COLORMAP
