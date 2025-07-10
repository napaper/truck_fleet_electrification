import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from fpdf import FPDF
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots
import warnings

# Add latex binary to python path to use scienceplots style 
# see https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-latex for more information
# the following line is for MacOS, for other operating systems the path might be different
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'


from IPython.display import Markdown as md
from smvis.gridfigure import GridFigure
from smvis.utils import genLineLegendHandle, setFontSize, setLatexRendering

try:
    # Attempt relative import if within a package
    from . import data_processing as dp
except ImportError:
    # Fall back to absolute import if running as a script
    import data_processing as dp
    print('relative import failed, using absolute import instead')

# HERE optionally set the style to scienceplots
plt.style.use(['science', 'nature'])
#plt.style.use(['default'])

#plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.size'] = 9
plt.rcParams['legend.frameon'] = True

# Define alpha values for grid lines
alpha_major = dp.alpha_major
alpha_minor = dp.alpha_minor

# Colors definition
colors = {
    'TUMBlack': '#000000',
    'TUMWhite': '#FFFFFF',
    'TUMBlue1': '#005293',
    'TUMOrange': '#E37222', # 2
    'TUMBlue2': '#3070b3',  # 3
    'TUMGreen3': '#A2AD00', # 4 (light green)
    'TUMBlue3': '#64A0C8',
    'TUMGreen2': '#515600', # 6 (dark green)
    'TUMBlue4': '#98C6EA',  # light blue
    'TUMGray1': '#333333',  # dark 
    'TUMBlue5': '#B0D0F0',
    'TUMGray2': '#7F7F7F',  # 10 medium
    'TUMBlue6': '#2A4D70',
    'TUMGray3': '#CCCCCC',  # light
    'TUMBlueDark': '#0B1340',
    'TUMIvory': '#DAD7CB',
    'TUMGreen1': '#292B00',
    'Purple': '#B3679B',    # 15
    'LightPurple': "#020202",
    'DarkPurple': '#31081F',
    'LightYellow': '#FFE548', #18
    'Rust': '#885053'         #19
}

# Remove white and light gray colors from the palette
line_colors = {k: v for k, v in colors.items() if v not in ['#FFFFFF', '#CCCCCC']}

# Function to calculate brightness
def brightness(hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    return hls[1]  # Luminance value

# Sort color_list by brightness
color_list = list(line_colors.values())
color_list_sorted = sorted(color_list, key=brightness, reverse=True)

colors_plot = list(color_list[i] for i in [10, 2, 3, 4, 15, 18, 19, 1, 6])

cmap = ListedColormap(color_list_sorted[:97])


# -------------------------------------------------------------------------------------------
#                              DATA RECORDING - WEEKLY DISTANCES
# -------------------------------------------------------------------------------------------


# Suppress the specific UserWarning about timezone info being dropped +TODO: remove this once the issue is resolved in pandas
warnings.filterwarnings("ignore", category=UserWarning, message="Converting to PeriodArray/Index representation will drop timezone information.")
warnings.filterwarnings("ignore", category=UserWarning, message="Converting to Period representation will drop timezone information.")


def plot_weekly_distances(df_trips): #TODO: rename to plot_weekly_distances and delete old version in code
    """
    Plot the weekly distances covered by the fleet as a stacked bar plot, centered on week start dates (Sunday).
    """
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = textwidth * 1.8  # Height remains increased
    width = textwidth * 2.5  # Increased width for larger date labels

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, h_169))

    # Preprocess
    df_trips['start_time_7d'] = pd.to_datetime(df_trips['start_time'], utc=True).dt.to_period("W-SUN")
    distances_km = df_trips.groupby(['start_time_7d', 'freight_forwarder'])['distance'].sum().unstack(fill_value=0) / 1000

    # Get first and last recording date for each fleet
    fleet_periods = df_trips.groupby('freight_forwarder').agg(
        first_date=('start_time', 'min'),
        last_date=('start_time', 'max')
    )

    # Create a color mapping for freight forwarders
    freight_forwarders = sorted(df_trips['freight_forwarder'].unique())
    colors_plot = ['#1f77b4', '#2c2c2c', '#ff7f0e', '#a0c114', '#7F7F7F', '#B3679B']
    ff_colors = {}
    for i, ff in enumerate(freight_forwarders):
        ff_colors[ff] = colors_plot[i % len(colors_plot)]

    # Plot
    distances_km.plot.bar(stacked=True, xlabel='', ylabel='distance / km', ax=ax, color=[ff_colors.get(i, '#808080') for i in distances_km.columns])

    # Configure the plot
    ax.set_ylim(0, 82000)  # Capped at 85,000 for y-axis
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5000))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t, pos: f"{int(t):,}"))
    ax.set_ylabel('distance / km', fontsize=14)
    ax.set_xlabel('week', fontsize=14)

    # X-axis ticks (show all weeks for readability with larger font)
    x_week_ticks = np.arange(len(distances_km))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(x_week_ticks))
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda t, pos: f"{distances_km.index[t].start_time.day:02d}-{distances_km.index[t].start_time.month:02d}-{distances_km.index[t].start_time.year}"))
    
    # Remove major year ticks to avoid clutter, since we add dividers
    ax.xaxis.set_major_locator(ticker.NullLocator())

    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", which="minor", pad=20, labelrotation=90, labelsize=11)  
    ax.grid(axis="y", which='major', alpha=0.3)
    ax.grid(axis="y", which='minor', alpha=0.15)

    # Add horizontal lines for fleet recording periods
    indicator_level = 70000
    max_fleet = max(fleet_periods.index) if not fleet_periods.empty else 6  
    for i, (fleet, (first_date, last_date)) in enumerate(fleet_periods.iterrows()):
        first_week = pd.Timestamp(first_date).to_period("W-SUN").start_time
        last_week = pd.Timestamp(last_date).to_period("W-SUN").start_time
        first_idx = np.searchsorted(distances_km.index.to_timestamp(), first_week)
        last_idx = np.searchsorted(distances_km.index.to_timestamp(), last_week)
        ax.plot([first_idx, last_idx], [indicator_level + i * 2000, indicator_level + i * 2000], 
                '--|', linewidth=1, markersize=7, color=ff_colors.get(fleet, 'gray'))

    # Create year divider for 2021-2022
    year_change_idx_2021_2022 = np.searchsorted(distances_km.index.to_timestamp(), pd.Timestamp('2022-01-01').to_period("W-SUN").start_time)
    ax.axvline(x=year_change_idx_2021_2022 - 0.5, color='black', linestyle='--', alpha=0.5)

    # Create year divider for 2022-2023
    year_change_idx_2022_2023 = np.searchsorted(distances_km.index.to_timestamp(), pd.Timestamp('2023-01-01').to_period("W-SUN").start_time)
    ax.axvline(x=year_change_idx_2022_2023 - 0.5, color='black', linestyle='--', alpha=0.5)

    # Add legend (moved to upper right, entries listed vertically, kept within plot)
    ax.legend(ncol=1, title="fleet", loc='upper right', bbox_to_anchor=(1.0,0.9),fontsize=13, title_fontsize=14)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, right=0.80)  

    # Save the plot
    plt.savefig('data/output/figures/operational/data_recordings.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/data_recordings.pdf', bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------------------------
#              KDE PLOTS - TRACK - DISTANCE | DURATION | MAX SPEED | AVG SPEED
# -------------------------------------------------------------------------------------------


# Suppress the specific UserWarning about timezone info being dropped +TODO: remove this once the issue is resolved in pandas
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has 0 variance; skipping density estimate.")


def plot_kde_plots(df_trips):
    """
    Plot four KDE plots in a 2x2 grid to show the distribution of the data with vehicle IDs coloring.
    Vehicles from the same freight forwarder will share the same color.
    """
    textwidth = 159.2 / 25.4  # Umrechnung der Textbreite von mm in Zoll
    height = textwidth * 1.2  # Etwas höhere Höhe
    width = textwidth * 1.5  # Größere Breite
    
    fig, axes = plt.subplots(2, 2, figsize=(width, height), sharey=False)

    # Group vehicle IDs by freight forwarder
    vehicle_to_ff = df_trips[['vehicle_id', 'freight_forwarder']].drop_duplicates().set_index('vehicle_id')['freight_forwarder']
    freight_forwarders = sorted(df_trips['freight_forwarder'].unique())
    
    # Create a color mapping for freight forwarders
    ff_colors = {}
    for i, ff in enumerate(freight_forwarders):
        ff_colors[ff] = colors_plot[i % len(colors_plot)]  # Loop through the colors if there are more FFs than colors
    
    # Create palette that assigns the same color to all vehicles from the same freight forwarder
    palette = {vehicle_id: ff_colors[vehicle_to_ff.loc[vehicle_id]] for vehicle_id in df_trips['vehicle_id'].unique()}

    # Create a mapping for the colorbar - we'll map vehicle IDs to their FF colors but maintain numeric ordering
    n_vehicles = df_trips['vehicle_id'].nunique()
    norm = Normalize(vmin=1, vmax=n_vehicles)
    vehicle_ids = sorted(df_trips['vehicle_id'].unique())
    cmap_colors = [ff_colors[vehicle_to_ff.loc[vid]] for vid in vehicle_ids]
    custom_cmap = ListedColormap(cmap_colors)

    sns.kdeplot(data=df_trips, x='distance_km', hue='vehicle_id', ax=axes[0, 0], palette=palette, common_norm=False, fill=True)
    axes[0, 0].set_xlabel('Distance / km')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend_.remove()
    axes[0, 0].set_xlim(0, 125)
    mean_distance = df_trips['distance_km'].mean()
    std_distance = df_trips['distance_km'].std()
    median_distance = df_trips['distance_km'].median()
    axes[0, 0].axvline(mean_distance, color='r', linestyle='--')
    axes[0, 0].axvline(mean_distance + std_distance, color='r', linestyle=':')
    axes[0, 0].axvline(mean_distance - std_distance, color='r', linestyle=':')
    axes[0, 0].axvline(median_distance, color='orange', linestyle='-')
    # Add labels in top right corner
    axes[0, 0].text(0.95, 0.95, f'Mean: {mean_distance:.1f} km\nMedian: {median_distance:.1f} km', 
                   transform=axes[0, 0].transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    # Add grid
    axes[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    sns.kdeplot(data=df_trips, x='duration_h', hue='vehicle_id', ax=axes[0, 1], palette=palette, common_norm=False, fill=True)
    axes[0, 1].set_xlabel('Duration / h')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend_.remove()
    axes[0, 1].set_xlim(0, 4)
    mean_duration = df_trips['duration_h'].mean()
    std_duration = df_trips['duration_h'].std()
    median_duration = df_trips['duration_h'].median()
    axes[0, 1].axvline(mean_duration, color='r', linestyle='--')
    axes[0, 1].axvline(mean_duration + std_duration, color='r', linestyle=':')
    axes[0, 1].axvline(mean_duration - std_duration, color='r', linestyle=':')
    axes[0, 1].axvline(median_duration, color='orange', linestyle='-')
    # Add labels in top right corner
    axes[0, 1].text(0.95, 0.95, f'Mean: {mean_duration:.2f} h\nMedian: {median_duration:.2f} h', 
                   transform=axes[0, 1].transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    # Add grid
    axes[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    sns.kdeplot(data=df_trips, x='max_speed_kmh', hue='vehicle_id', ax=axes[1, 0], palette=palette, common_norm=False, fill=True)
    axes[1, 0].set_xlabel('Max Speed / km/h')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend_.remove()
    axes[1, 0].set_xlim(0, 150)
    mean_max_speed = df_trips['max_speed_kmh'].mean()
    std_max_speed = df_trips['max_speed_kmh'].std()
    median_max_speed = df_trips['max_speed_kmh'].median()
    axes[1, 0].axvline(mean_max_speed, color='r', linestyle='--')
    axes[1, 0].axvline(mean_max_speed + std_max_speed, color='r', linestyle=':')
    axes[1, 0].axvline(mean_max_speed - std_max_speed, color='r', linestyle=':')
    axes[1, 0].axvline(median_max_speed, color='orange', linestyle='-')
    # Add labels in top right corner
    axes[1, 0].text(0.95, 0.95, f'Mean: {mean_max_speed:.1f} km/h\nMedian: {median_max_speed:.1f} km/h', 
                   transform=axes[1, 0].transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    # Add grid
    axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    sns.kdeplot(data=df_trips, x='avg_speed_kmh', hue='vehicle_id', ax=axes[1, 1], palette=palette, common_norm=False, fill=True)
    axes[1, 1].set_xlabel('Average Speed / km/h')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend_.remove()
    axes[1, 1].set_xlim(0, 100)
    mean_avg_speed = df_trips['avg_speed_kmh'].mean()
    std_avg_speed = df_trips['avg_speed_kmh'].std()
    median_avg_speed = df_trips['avg_speed_kmh'].median()
    axes[1, 1].axvline(mean_avg_speed, color='r', linestyle='--')
    axes[1, 1].axvline(mean_avg_speed + std_avg_speed, color='r', linestyle=':')
    axes[1, 1].axvline(mean_avg_speed - std_avg_speed, color='r', linestyle=':')
    axes[1, 1].axvline(median_avg_speed, color='orange', linestyle='-')
    # Add labels in top right corner
    axes[1, 1].text(0.95, 0.95, f'Mean: {mean_avg_speed:.1f} km/h\nMedian: {median_avg_speed:.1f} km/h', 
                   transform=axes[1, 1].transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    # Add grid
    axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout to make room for the single legend on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Add a single color bar on the right showing vehicle IDs but colored by their freight forwarder
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, aspect=30, shrink=0.8)
    cbar.ax.invert_yaxis()
    
    # Create a legend for the lines (mean, std dev, median) that appears above the colorbar
    red_lines = [plt.Line2D([0], [0], color='r', linestyle='--', label='Mean'),
                 plt.Line2D([0], [0], color='r', linestyle=':', label='1 Std. Dev.'),
                 plt.Line2D([0], [0], color='orange', linestyle='-', label='Median')]
    
    # Add the legend directly above the colorbar
    fig.legend(handles=red_lines, loc='upper right', bbox_to_anchor=(0.95, 0.99), 
              fontsize=9, frameon=True, title="Statistics")
    
    # Add freight forwarder legend to explain the colors
    handles = [mpatches.Patch(color=color, label=f'FF {ff}') for ff, color in ff_colors.items()]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.222), 
              fontsize=9, frameon=True, title="Freight Forwarders")
    
    cbar.set_label('Vehicle IDs', labelpad=15, fontsize=12)
    
    plt.savefig('data/output/figures/operational/kde_plots.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/kde_plots.pdf', bbox_inches='tight')

    plt.show()


# -------------------------------------------------------------------------------------------
#                KDE PLOTS - TOUR - DISTANCE | DURATION | MAX SPEED | AVG SPEED
# -------------------------------------------------------------------------------------------


def plot_tour_kde_plots(df_trips):
    """
    Plot two KDE plots in a 1x2 grid to show the distribution of cumulative distances and durations by tour.
    Tours are colored by freight forwarder.
    
    Parameters:
    -----------
    df_trips : pandas DataFrame
        DataFrame containing trip data with tour_id, vehicle_id, freight_forwarder, distance_km, and duration_h columns
    """
    # First, calculate cumulative values by tour_id
    tour_aggregates = df_trips.groupby(['tour_id', 'freight_forwarder', 'vehicle_id']).agg({
        'distance_km': 'sum',  # Sum of all distances in the tour
        'duration_h': 'sum'    # Sum of all durations in the tour
    }).reset_index()
    
    # Set figure dimensions
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    height = textwidth * 0.8   # Increase height to accommodate legends at the top
    
    # Create figure with more space at the top for legends
    fig = plt.figure(figsize=(textwidth, height))
    
    # Create gridspec for more control over subplot placement
    gs = fig.add_gridspec(2, 2, height_ratios=[0.2, 0.8])
    
    # Create a row for legends at the top
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis('off')  # Hide axes for the legend area
    
    # Create the two plot axes in the second row
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2]
    
    # Create a color mapping for freight forwarders
    freight_forwarders = sorted(tour_aggregates['freight_forwarder'].unique())
    ff_colors = {}
    for i, ff in enumerate(freight_forwarders):
        ff_colors[ff] = colors_plot[i % len(colors_plot)]
    
    # Create palette that assigns colors based on freight forwarder
    palette = {vehicle_id: ff_colors[ff] for vehicle_id, ff in 
              tour_aggregates[['vehicle_id', 'freight_forwarder']].drop_duplicates().values}
    
    # Plot distance KDE
    sns.kdeplot(data=tour_aggregates, x='distance_km', hue='vehicle_id', 
                ax=axes[0], palette=palette, common_norm=False, fill=True)
    axes[0].set_xlabel('Cumulative Tour Distance / km')
    axes[0].set_ylabel('Density')
    axes[0].legend_.remove()
    
    # Calculate and show statistics for distance
    mean_distance = tour_aggregates['distance_km'].mean()
    std_distance = tour_aggregates['distance_km'].std()
    median_distance = tour_aggregates['distance_km'].median()
    
    # Set reasonable x-limit based on data
    max_xlim_distance = min(800, tour_aggregates['distance_km'].quantile(0.99))
    axes[0].set_xlim(0, max_xlim_distance)
    
    # Add reference lines
    axes[0].axvline(mean_distance, color='r', linestyle='--')
    axes[0].axvline(mean_distance + std_distance, color='r', linestyle=':')
    axes[0].axvline(mean_distance - std_distance, color='r', linestyle=':')
    axes[0].axvline(median_distance, color='orange', linestyle='-')
    
    # Add statistics box
    axes[0].text(0.95, 0.95, f'Mean: {mean_distance:.1f} km\nMedian: {median_distance:.1f} km', 
               transform=axes[0].transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    
    # Add grid
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot duration KDE
    sns.kdeplot(data=tour_aggregates, x='duration_h', hue='vehicle_id', 
                ax=axes[1], palette=palette, common_norm=False, fill=True)
    axes[1].set_xlabel('Cumulative Tour Duration / h')
    axes[1].set_ylabel('Density')
    axes[1].legend_.remove()
    
    # Calculate and show statistics for duration
    mean_duration = tour_aggregates['duration_h'].mean()
    std_duration = tour_aggregates['duration_h'].std()
    median_duration = tour_aggregates['duration_h'].median()
    
    # Set reasonable x-limit based on data
    max_xlim_duration = min(30, tour_aggregates['duration_h'].quantile(0.99))
    axes[1].set_xlim(0, max_xlim_duration)
    
    # Add reference lines
    axes[1].axvline(mean_duration, color='r', linestyle='--')
    axes[1].axvline(mean_duration + std_duration, color='r', linestyle=':')
    axes[1].axvline(mean_duration - std_duration, color='r', linestyle=':')
    axes[1].axvline(median_duration, color='orange', linestyle='-')
    
    # Add statistics box
    axes[1].text(0.95, 0.95, f'Mean: {mean_duration:.2f} h\nMedian: {median_duration:.2f} h', 
               transform=axes[1].transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    
    # Add grid
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Create a legend for the statistics
    red_lines = [plt.Line2D([0], [0], color='r', linestyle='--', label='Mean'),
                 plt.Line2D([0], [0], color='r', linestyle=':', label='1 Std. Dev.'),
                 plt.Line2D([0], [0], color='orange', linestyle='-', label='Median')]
    
    # Create freight forwarder legend handles
    ff_handles = [mpatches.Patch(color=color, label=f'FF {ff}') for ff, color in ff_colors.items()]
    
    # Add legends to the top area
    # First legend: Statistics (placed on the left side)
    stats_legend = legend_ax.legend(handles=red_lines, loc='upper left', 
                                    bbox_to_anchor=(0.01, 0.9),
                                    fontsize=9, frameon=True, title="Statistics")
    legend_ax.add_artist(stats_legend)  # Add the first legend
    
    # Second legend: Freight forwarders (placed on the right side)
    # Adjust ncol to display in 3x2 grid (3 columns, 2 rows if there are 6 freight forwarders)
    ncols = min(3, len(ff_colors))
    ff_legend = legend_ax.legend(handles=ff_handles, loc='upper right', 
                                bbox_to_anchor=(0.99, 0.9),
                                fontsize=9, frameon=True, title="Freight Forwarders",
                                ncol=ncols)  # 3 columns
    
    plt.tight_layout()
    plt.savefig('data/output/figures/operational/tour_kde_plots.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/tour_kde_plots.pdf', bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------------------------
#                                WEEKLY DISTANCE BOXPLOTS
# -------------------------------------------------------------------------------------------


def verify_weekday_aggregation(df_trips):
    print(plt.style)
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['weekday'] = df_trips['start_time'].dt.weekday  # 0 = Montag, 6 = Sonntag
    df_trips['date'] = df_trips['start_time'].dt.date
    daily_distance = df_trips.groupby(['vehicle_id', 'date', 'weekday'])['distance_km'].sum().reset_index()

    print("Verifying weekday aggregation:")
    for weekday in range(7):  # Überprüfen aller Wochentage
        print(f"Sample data for {'Monday' if weekday == 0 else 'Tuesday' if weekday == 1 else 'Wednesday' if weekday == 2 else 'Thursday' if weekday == 3 else 'Friday' if weekday == 4 else 'Saturday' if weekday == 5 else 'Sunday'}:")
        sample_data = daily_distance[daily_distance['weekday'] == weekday].sample(5, random_state=1)  # Zeigt 5 zufällige Stichproben
        print(sample_data)


def plot_weekly_distance_boxplot(df_trips):
    """
    Creates a figure with a central boxplot showing all freight forwarders' data together
    at the top, and individual plots for each freight forwarder below.
    
    Parameters:
    -----------
    df_trips : DataFrame
        DataFrame containing trip data with vehicle_id, start_time, distance_km, and freight_forwarder columns
    """
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['weekday'] = df_trips['start_time'].dt.weekday  # 0 = Monday, 6 = Sunday
    df_trips['date'] = df_trips['start_time'].dt.date
    daily_distance = df_trips.groupby(['vehicle_id', 'date', 'weekday', 'freight_forwarder'])['distance_km'].sum().reset_index()

    # Define plot size
    textwidth = (159.2 / 25.4)*2  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    height = h_169 * 2  # Increase height for better readability
    width = textwidth * 1.2  # Increase width for better layout

    # Define colors for weekdays and average
    weekday_colors = [
        colors['TUMBlue4'], colors['TUMBlue3'], colors['TUMBlue2'], 
        colors['TUMBlue1'], colors['TUMGray3'], colors['TUMGray2'], 
        colors['TUMGray1']
    ]
    avg_color = colors['LightPurple']  # Special color for the average column

    # Create subplots for each freight forwarder plus one for the central plot
    fig = plt.figure(figsize=(width, height))
    
    # Add a GridSpec with 4 rows and 2 columns
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # Add the central plot at the top, spanning both columns
    central_ax = fig.add_subplot(gs[0, :])
    
    # Create the individual freight forwarder axes
    axes = []
    for i in range(1, 4):  # Rows 1-3
        for j in range(2):  # 2 columns
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Get unique freight forwarders
    freight_forwarders = sorted(df_trips['freight_forwarder'].unique())
    
    # Prepare data for the central plot
    all_boxplot_data = []
    for weekday in range(7):
        weekday_data = daily_distance[daily_distance['weekday'] == weekday]['distance_km'].tolist()
        all_boxplot_data.append(weekday_data)
    
    # Add the average data for the central plot (all weekdays combined)
    all_avg_data = daily_distance['distance_km'].tolist()
    all_boxplot_data.append(all_avg_data)
    
    # Create central boxplot with all data
    scaling_factor = textwidth / 13
    bp_central = central_ax.boxplot(
        all_boxplot_data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
        medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
        showfliers=True,
        flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
        whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
        capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
        widths=0.55 * scaling_factor,
        positions=range(len(all_boxplot_data)),
        labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Combined']
    )
    
    # Style central boxplot
    central_ax.set_title('All Freight Forwarders Combined', fontsize=14 * scaling_factor)
    central_ax.tick_params(axis='x', labelsize=12 * scaling_factor)
    central_ax.tick_params(axis='y', labelsize=12 * scaling_factor)
    central_ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
    central_ax.set_ylim(bottom=0)
    
    # Assign colors to the central boxplots (7 weekdays + average)
    for i, (box, color) in enumerate(zip(bp_central['boxes'], weekday_colors + [avg_color])):
        box.set_facecolor(color)
        # Add a slightly thicker border to the average box to make it stand out
        if i == 7:  # Average box
            box.set(linewidth=2.0 * scaling_factor)
    
    # Annotate central plot median values - moved to the left to overlap with the median line
    for n, (median_feature, distances) in enumerate(zip(bp_central['medians'], all_boxplot_data)):
        distances_series = pd.Series(distances)
        median_value = distances_series.median()
        x_median, y_median = median_feature.get_xydata()[1]
        # Move text position more to the left (was +0.25, now 0.0 to center on the median line)
        central_ax.text(x_median, y_median, f'{median_value:.2f}', 
                horizontalalignment='center', color=colors['TUMBlack'], fontsize=10 * scaling_factor, 
                bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=colors['TUMWhite'], 
                        edgecolor=colors['TUMOrange'], alpha=0.9))
    
    # Process each freight forwarder for individual plots
    for i, ff in enumerate(freight_forwarders):
        if i >= len(axes):
            break
            
        # Filter data for current freight forwarder
        ff_data = daily_distance[daily_distance['freight_forwarder'] == ff]
        
        # Calculate and print daily distance per truck for the current freight forwarder
        print(f"\nFreight Forwarder {ff} - Daily Distance per Truck:")
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = ff_data.groupby('weekday')['distance_km'].agg(['mean', 'std', 'median', 'count']).reset_index()
        daily_stats['weekday_name'] = daily_stats['weekday'].apply(lambda x: weekday_names[x])
        print(daily_stats[['weekday_name', 'mean', 'std', 'median', 'count']])
        print(f"Overall mean: {ff_data['distance_km'].mean():.2f} km")
                
        # Group by weekday and prepare boxplot data
        boxplot_data = ff_data.groupby('weekday')['distance_km'].apply(list)
        
        # Add the average data for this freight forwarder (all weekdays combined)
        avg_data = ff_data['distance_km'].tolist()
        boxplot_data_list = list(boxplot_data) + [avg_data]
        
        # Create boxplot
        bp = axes[i].boxplot(
            boxplot_data_list,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
            medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
            showfliers=True,
            flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
            whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            widths=0.55 * scaling_factor,
            positions=range(len(boxplot_data_list)),
            labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Combined']
        )
        
        # Set x-axis and y-axis tick labels font size
        axes[i].tick_params(axis='x', labelsize=12 * scaling_factor)
        axes[i].tick_params(axis='y', labelsize=12 * scaling_factor)
        
        # Annotate median values - moved to the left to overlap with the median line
        for n, (median_feature, distances) in enumerate(zip(bp['medians'], boxplot_data_list)):
            distances_series = pd.Series(distances)
            median_value = distances_series.median()
            x_median, y_median = median_feature.get_xydata()[1]
            # Move text position more to the left (was +0.25, now 0.0 to center on the median line)
            axes[i].text(x_median, y_median, f'{median_value:.2f}', 
                    horizontalalignment='center', color=colors['TUMBlack'], fontsize=10 * scaling_factor, 
                    bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=colors['TUMWhite'], 
                            edgecolor=colors['TUMOrange'], alpha=0.9))
        
        # Assign colors to the boxplots
        for j, (box, color) in enumerate(zip(bp['boxes'], weekday_colors + [avg_color])):
            box.set_facecolor(color)
            # Add a slightly thicker border to the average box to make it stand out
            if j == 7:  # Average box
                box.set(linewidth=2.0 * scaling_factor)
        
        # Set title for each subplot
        axes[i].set_title(f'Freight Forwarder {ff}', fontsize=12 * scaling_factor)
        
        # Add grid
        axes[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
        
        # Set y-axis limit
        axes[i].set_ylim(bottom=0)
        
        # Customize spines
        for spine in axes[i].spines.values():
            spine.set_edgecolor(colors['TUMBlack'])
            spine.set_linewidth(0.8 * scaling_factor)
    
    # Remove any empty subplots if there are fewer than 6 freight forwarders
    for i in range(len(freight_forwarders), len(axes)):
        fig.delaxes(axes[i])
    
    # Add common labels
    fig.text(0.5, 0.01, 'Weekday', ha='center', va='center', fontsize=16 * scaling_factor)
    fig.text(0.01, 0.5, 'Distance per Truck / km', ha='center', va='center', rotation='vertical', fontsize=16 * scaling_factor)
    
    # Define legend handle
    mean_line = mlines.Line2D([], [], color=colors['TUMGreen3'], linestyle='--', label='Mean', linewidth=3 * scaling_factor)
    median_line = mlines.Line2D([], [], color=colors['TUMOrange'], label='Median', linewidth=3 * scaling_factor)
    fig.legend(handles=[median_line, mean_line], loc='upper center', fontsize=14 * scaling_factor, frameon=True, ncol=2)

    # Remove any empty subplots if there are fewer than 6 freight forwarders
    for i in range(len(freight_forwarders), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])  # Adjust layout to make room for common labels
    plt.savefig('data/output/figures/operational/freight_forwarder_weekday_boxplots.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/freight_forwarder_weekday_boxplots.pdf', bbox_inches='tight')

    plt.show()


# -------------------------------------------------------------------------------------------
#                                Tour DURATION & DISTANCE ECDF
# -------------------------------------------------------------------------------------------


def plot_tour_duration_distance_ecdf(df_trips, max_distance=800, max_duration=30):
    """
    Plot empirical cumulative distribution functions (ECDF) for cumulative tour durations and distances.
    Shows the percentage of tours below certain thresholds for each freight forwarder.
    
    Parameters:
    -----------
    df_trips : pandas DataFrame
        DataFrame containing trip data with tour_id, freight_forwarder, distance_km, and duration_h columns
    max_distance : float, optional
        Maximum distance to display in km (only affects plot display, not statistics calculation)
    max_duration : float, optional
        Maximum duration to display in hours (only affects plot display, not statistics calculation)
    """
    # First, calculate cumulative values by tour_id
    tour_aggregates = df_trips.groupby(['tour_id', 'freight_forwarder', 'vehicle_id']).agg({
        'distance_km': 'sum',  # Sum of all distances in the tour
        'duration_h': 'sum'    # Sum of all durations in the tour
    }).reset_index()
    
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    width = textwidth * 1.2
    
    # Create figure with GridSpec for better control over subplot placement
    fig = plt.figure(figsize=(width, h_169))
    gs = plt.GridSpec(2, 1, height_ratios=[0.1, 0.9])
    
    # Create area for legend at the top
    legend_ax = fig.add_subplot(gs[0])
    legend_ax.axis('off')  # Hide axes for the legend area
    
    # Create main plotting area with two side-by-side subplots
    plot_gs = gs[1].subgridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(plot_gs[0])
    ax2 = fig.add_subplot(plot_gs[1])
    
    # Create a color mapping for freight forwarders
    freight_forwarders = sorted(tour_aggregates['freight_forwarder'].unique())
    ff_colors = {}
    for i, ff in enumerate(freight_forwarders):
        ff_colors[ff] = colors_plot[i % len(colors_plot)]
    
    # Keep track of handles for the legend
    handles = []
    labels = []
    
    # Create ECDF plots for each freight forwarder
    for ff in freight_forwarders:
        ff_data = tour_aggregates[tour_aggregates['freight_forwarder'] == ff]
        
        # Duration ECDF
        sns.ecdfplot(
            data=ff_data, 
            x='duration_h',
            color=ff_colors[ff],
            ax=ax1,
            linewidth=2,
            label=f'FF {ff}'
        )
        
        # Distance ECDF
        sns.ecdfplot(
            data=ff_data, 
            x='distance_km',
            color=ff_colors[ff],
            ax=ax2,
            linewidth=2
        )
        
        # Add handle for legend (only need to do this once per freight forwarder)
        handles.append(plt.Line2D([0], [0], color=ff_colors[ff], lw=2))
        labels.append(f'FF {ff}')
    
    # Set titles and labels
    ax1.set_title('Tour Duration')
    ax2.set_title('Tour Distance')
    
    # Configure tick locations
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    
    # Add grid lines
    ax1.grid(which='major', alpha=alpha_major)
    ax2.grid(which='major', alpha=alpha_major)
    ax1.grid(which='minor', alpha=alpha_minor)
    ax2.grid(which='minor', alpha=alpha_minor)
    
    # Set axis limits and labels with max limits
    ax1.set(xlim=(0, max_duration), xlabel='Tour Duration / h', ylabel='Cumulative Probability')
    ax2.set(xlim=(0, max_distance), xlabel='Tour Distance / km', ylabel='')
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    
    # Remove individual legends from subplots
    for ax in [ax1, ax2]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add reference lines at key percentiles (25%, 50%, 75%) - calculated from COMPLETE dataset
    percentiles = [0.25, 0.5, 0.75]
    linestyles = [':', '--', '-.']
    
    for p, ls in zip(percentiles, linestyles):
        ax1.axhline(y=p, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
        ax2.axhline(y=p, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
        
        # Add percentile values for the complete dataset (not filtered by max limits)
        duration_percentile = tour_aggregates['duration_h'].quantile(p)
        distance_percentile = tour_aggregates['distance_km'].quantile(p)
        
        # Only show vertical lines if they fall within the display range
        if duration_percentile <= max_duration:
            ax1.axvline(x=duration_percentile, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
        if distance_percentile <= max_distance:
            ax2.axvline(x=distance_percentile, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
    
    # Add a single legend at the top of the figure
    legend_ax.legend(handles=handles, labels=labels, ncol=len(handles), loc='center', 
                     title='Freight Forwarder', fontsize=8, title_fontsize=8)
    
    # Set consistent font size
    setFontSize([ax1, ax2], 9)
    
    # Add statistics annotations - calculated from COMPLETE dataset
    for i, (label, ax) in enumerate([('duration_h', ax1), ('distance_km', ax2)]):
        full_data = tour_aggregates[label]
        median_val = full_data.median()
        q75_val = full_data.quantile(0.75)
        unit = 'h' if i == 0 else 'km'
        
        # Add text with key percentiles in the lower right corner
        ax.text(0.95, 0.05, f'Median: {median_val:.1f} {unit}\n75th percentile: {q75_val:.1f} {unit}', 
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    
    plt.savefig('data/output/figures/operational/tour_duration_and_distance_ecdf.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/tour_duration_and_distance_ecdf.pdf', bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------------------------
#                              TOUR DURATION & DISTANCE HISTOGRAMS
# -------------------------------------------------------------------------------------------


def plot_tour_duration_distance_histogram(df_trips, max_distance=800, max_duration=30):
    """
    Plot histograms for cumulative tour durations and distances with lines per freight forwarder.
    Tours are shown as percentage of total tours for each freight forwarder.
    
    Parameters:
    -----------
    df_trips : pandas DataFrame
        DataFrame containing trip data with tour_id, freight_forwarder, distance_km, and duration_h columns
    max_distance : float, optional
        Maximum distance to display in km (only affects plot display, not statistics calculation)
    max_duration : float, optional
        Maximum duration to display in hours (only affects plot display, not statistics calculation)
    """
    # First, calculate cumulative values by tour_id
    tour_aggregates = df_trips.groupby(['tour_id', 'freight_forwarder', 'vehicle_id']).agg({
        'distance_km': 'sum',  # Sum of all distances in the tour
        'duration_h': 'sum'    # Sum of all durations in the tour
    }).reset_index()
    
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    width = textwidth * 1.2
    
    # Create figure with GridSpec for better control over subplot placement
    fig = plt.figure(figsize=(width, h_169))
    gs = plt.GridSpec(2, 1, height_ratios=[0.1, 0.9])
    
    # Create area for legend at the top
    legend_ax = fig.add_subplot(gs[0])
    legend_ax.axis('off')  # Hide axes for the legend area
    
    # Create main plotting area with two side-by-side subplots
    plot_gs = gs[1].subgridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(plot_gs[0])
    ax2 = fig.add_subplot(plot_gs[1])
    axes = [ax1, ax2]
    
    # Create a color mapping for freight forwarders
    freight_forwarders = sorted(tour_aggregates['freight_forwarder'].unique())
    ff_colors = {}
    for i, ff in enumerate(freight_forwarders):
        ff_colors[ff] = colors_plot[i % len(colors_plot)]
    
    # Create palette that assigns colors based on freight forwarder
    palette = {vehicle_id: ff_colors[ff] for vehicle_id, ff in 
              tour_aggregates[['vehicle_id', 'freight_forwarder']].drop_duplicates().values}
    
    # Create visualization data (filtered for display purposes)
    duration_data_viz = tour_aggregates[tour_aggregates['duration_h'] <= max_duration]
    distance_data_viz = tour_aggregates[tour_aggregates['distance_km'] <= max_distance]
    
    # Keep track of handles for the legend
    handles = []
    labels = []
    
    # Calculate total number of tours per freight forwarder for percentage calculation
    # Use the COMPLETE dataset for statistics
    ff_tour_counts = tour_aggregates.groupby('freight_forwarder').size().to_dict()
    
    # Create histograms for each freight forwarder
    for ff in freight_forwarders:
        # Get total number of tours for this freight forwarder
        total_ff_tours = ff_tour_counts[ff]
        
        # Duration histogram - use filtered data for visualization only
        ff_duration_data_viz = duration_data_viz[duration_data_viz['freight_forwarder'] == ff]
        # Calculate weights for percentage (each tour will count as 1/total tours * 100%)
        duration_weights = np.ones(len(ff_duration_data_viz)) / total_ff_tours * 100
        
        sns.histplot(
            data=ff_duration_data_viz, 
            x='duration_h',
            bins=40,
            element="step", 
            fill=False,
            color=ff_colors[ff],
            ax=axes[0],
            label=f'FF {ff}',
            weights=duration_weights  # Use weights for percentage
        )
        
        # Distance histogram - use filtered data for visualization only
        ff_distance_data_viz = distance_data_viz[distance_data_viz['freight_forwarder'] == ff]
        # Calculate weights for percentage
        distance_weights = np.ones(len(ff_distance_data_viz)) / total_ff_tours * 100
        
        sns.histplot(
            data=ff_distance_data_viz, 
            x='distance_km',
            bins=40,
            element="step", 
            fill=False,
            color=ff_colors[ff],
            ax=axes[1],
            weights=distance_weights  # Use weights for percentage
        )
        
        # Add handle for legend (only need to do this once per freight forwarder)
        handles.append(plt.Line2D([0], [0], color=ff_colors[ff], lw=2))
        labels.append(f'FF {ff}')
    
    # Set titles and labels
    axes[0].set_title('Tour Duration Distribution')
    axes[1].set_title('Tour Distance Distribution')
    
    # Configure tick locations
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(25))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(2))
    
    # Add grid lines
    axes[0].grid(which='major', alpha=alpha_major)
    axes[1].grid(which='major', alpha=alpha_major)
    
    # Set axis limits and labels
    axes[0].set(xlim=(0, max_duration), xlabel='Tour Duration / h', ylabel='% of Tours')
    axes[1].set(xlim=(0, max_distance), xlabel='Tour Distance / km', ylabel='')
    
    # Format y-axis labels as percentages
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    
    # Remove individual legends from subplots
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add a single legend at the top of the figure
    legend_ax.legend(handles=handles, labels=labels, ncol=len(handles), loc='center', 
                     title='Freight Forwarder', fontsize=8, title_fontsize=8)
    
    # Set consistent font size
    setFontSize(axes, 9)
    
    # Add statistics to the plot - calculated from the COMPLETE dataset
    for i, (label, ax) in enumerate([('duration_h', axes[0]), ('distance_km', axes[1])]):
        # Use the original, unfiltered data for statistics
        full_data = tour_aggregates[label]
        mean_val = full_data.mean()
        median_val = full_data.median()
        unit = 'h' if i == 0 else 'km'
        
        # Add text with statistics in the upper right corner
        ax.text(0.95, 0.95, f'Mean: {mean_val:.1f} {unit}\nMedian: {median_val:.1f} {unit}', 
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    
    plt.savefig('data/output/figures/operational/tour_duration_and_distance_hist.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/tour_duration_and_distance_hist.pdf', bbox_inches='tight')

    plt.show()   

    
# -------------------------------------------------------------------------------------------
#                           TRIP DURATION & DISTANCE HISTOGRAMS & CDF
# -------------------------------------------------------------------------------------------


def plot_trip_duration_distance_histogram_cdf(df_trips, max_distance=400, max_duration=5):
    """
    Plot cumulative distribution plots for trip durations and distances.
    
    Parameters:
    -----------
    df_trips : pandas.DataFrame
        DataFrame containing trip information with duration_h and distance_km columns
    max_distance : float, optional
        Maximum distance to display in km
    max_duration : float, optional
        Maximum duration to display in hours
    """
    # Set up figure dimensions based on standard text width
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    
    # Create a grid figure for the two plots
    gf_dis = GridFigure(1, 2, textwidth, h_169, wspace=0.25, constrained_layout=False)
    
    # Filter data to the specified maximum values
    duration_data = df_trips.loc[df_trips.duration_h <= max_duration, 'duration_h']
    distance_data = df_trips.loc[df_trips.distance_km <= max_distance, 'distance_km']
    
    # Calculate CDF values manually for more control
    # Duration CDF
    duration_sorted = np.sort(duration_data)
    duration_y = np.arange(1, len(duration_sorted) + 1) / len(duration_sorted) * 100  # Convert to percentage
    
    # Distance CDF
    distance_sorted = np.sort(distance_data)
    distance_y = np.arange(1, len(distance_sorted) + 1) / len(distance_sorted) * 100  # Convert to percentage
    
    # Plot the CDFs
    gf_dis.axes_list[0].plot(duration_sorted, duration_y, linewidth=2.5, color=colors['TUMBlue1'])
    gf_dis.axes_list[1].plot(distance_sorted, distance_y, linewidth=2.5, color=colors['TUMBlue1'])
    
    # Add grid lines
    gf_dis.axes_list[0].grid(which='major', alpha=alpha_major)
    gf_dis.axes_list[1].grid(which='major', alpha=alpha_major)
    gf_dis.axes_list[0].grid(which='minor', alpha=alpha_minor)
    gf_dis.axes_list[1].grid(which='minor', alpha=alpha_minor)
    
    # Set titles and labels
    gf_dis.axes_list[0].set_title('Cumulative Duration Distribution')
    gf_dis.axes_list[1].set_title('Cumulative Distance Distribution')
    
    # Configure axis ticks and labels
    gf_dis.axes_list[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    gf_dis.axes_list[1].xaxis.set_major_locator(ticker.MultipleLocator(100))
    gf_dis.axes_list[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    gf_dis.axes_list[1].xaxis.set_minor_locator(ticker.MultipleLocator(25))
    gf_dis.axes_list[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
    gf_dis.axes_list[1].yaxis.set_major_locator(ticker.MultipleLocator(20))
    gf_dis.axes_list[0].yaxis.set_minor_locator(ticker.MultipleLocator(5))
    gf_dis.axes_list[1].yaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    # Set axis limits and labels
    gf_dis.axes_list[0].set(xlim=(0, max_duration), ylim=(0, 100), 
                            xlabel='Duration / h', ylabel='Cumulative Percentage')
    gf_dis.axes_list[1].set(xlim=(0, max_distance), ylim=(0, 100), 
                            xlabel='Distance / km', ylabel='')
    
    # Format y-axis as percentage
    gf_dis.axes_list[0].yaxis.set_major_formatter(ticker.PercentFormatter(100, decimals=0))
    gf_dis.axes_list[1].yaxis.set_major_formatter(ticker.PercentFormatter(100, decimals=0))
    
    # Add horizontal markers at key percentiles (25%, 50%, 75%)
    percentiles = [25, 50, 75]
    linestyles = [':', '--', '-.']
    
    for p, ls in zip(percentiles, linestyles):
        # Duration plot
        duration_value = np.percentile(duration_data, p)
        gf_dis.axes_list[0].axhline(y=p, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
        gf_dis.axes_list[0].axvline(x=duration_value, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
        
        # Distance plot
        distance_value = np.percentile(distance_data, p)
        gf_dis.axes_list[1].axhline(y=p, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
        gf_dis.axes_list[1].axvline(x=distance_value, linestyle=ls, color='gray', alpha=0.7, linewidth=1)
    
    # Add median value annotations
    median_duration = np.median(duration_data)
    median_distance = np.median(distance_data)
    
    gf_dis.axes_list[0].text(median_duration + 0.1, 52, f'Median: {median_duration:.1f} h', 
                            fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    gf_dis.axes_list[1].text(median_distance + 20, 52, f'Median: {median_distance:.1f} km', 
                            fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Set font size
    setFontSize(gf_dis.axes_list, 9)
    
    # Save figure
    plt.savefig('data/output/figures/operational/trip_duration_and_distance_cdf.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/trip_duration_and_distance_cdf.pdf', bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------------------------
#                           SOC BY FREIGHT FORWARDER BOXPLOTS
# -------------------------------------------------------------------------------------------


def plot_soc_by_freight_forwarder2(df_tours_energy):
    """
    Create boxplots showing the distribution of minimum SoC values
    for each freight forwarder, with each freight forwarder in a separate plot.
    
    Values below -20% are shown in a separate small plot on the right side.
    Main boxplot is offset to the left to make room for the outlier plot.
    
    Parameters:
    -----------
    df_tours_energy : pandas.DataFrame
        DataFrame containing tour energy data with soc_min and freight_forwarder columns
    """
    df = df_tours_energy.copy()
    
    # Convert freight_forwarder to string for consistent plotting
    df['freight_forwarder'] = df['freight_forwarder'].astype(str)
    
    # Get unique freight forwarders
    freight_forwarders = sorted(df['freight_forwarder'].unique())
    
    # Define plot size
    textwidth = (159.2 / 25.4)  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    height = h_169 * 1.5  # Increase height for better readability
    width = textwidth * 1.8  # Increase width for better layout
    
    # Define cutoff for extreme outliers
    cutoff = -0.2  # -20% SoC
    
    # Create figures with 2 plots each
    for fig_num in range((len(freight_forwarders) + 1) // 2):
        # Create a new figure for each pair of freight forwarders
        fig = plt.figure(figsize=(width, height))
        
        # Create GridSpec for layout control - 1 row, 2 columns for freight forwarders
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        
        # Process 2 freight forwarders per figure
        for i in range(2):
            ff_idx = fig_num * 2 + i
            
            # Check if we still have freight forwarders to process
            if ff_idx >= len(freight_forwarders):
                continue
                
            ff = freight_forwarders[ff_idx]
            
            # Filter data for current freight forwarder
            ff_data = df[df['freight_forwarder'] == ff]
            
            # Prepare min SoC data
            min_soc_data = ff_data['soc_min']
            
            # Split data into main and outliers
            main_data = min_soc_data[min_soc_data >= cutoff]
            outlier_data = min_soc_data[min_soc_data < cutoff]
            
            # Calculate and print SoC statistics for the current freight forwarder
            print(f"\nFreight Forwarder {ff} - Min SoC Statistics:")
            print(f"Mean={min_soc_data.mean():.2%}, Median={min_soc_data.median():.2%}")
            print(f"Min={min_soc_data.min():.2%}, Max={min_soc_data.max():.2%}")
            print(f"Number of extreme outliers (< {cutoff:.0%}): {len(outlier_data)}")
            
            # Create nested GridSpec similar to the image - main plot takes most space, outlier is small in corner
            gs_nested = gs[i].subgridspec(1, 4, width_ratios=[3, 1, 0.1, 0.8])
            
            # Main boxplot axis (use first 3/4 of space)
            main_ax = fig.add_subplot(gs_nested[0])
            
            # Small inset axis for extreme outliers (use last 1/4 of space)
            outlier_ax = fig.add_subplot(gs_nested[3])
            outlier_ax.set_facecolor('#ffeeee')  # Light pink background like in image
            
            # Add a border/frame around the outlier plot
            for spine in outlier_ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(1)
                
            # Scaling factor for consistent element sizes
            scaling_factor = textwidth / 10
            
            # Create boxplot for main data
            bp_main = main_ax.boxplot(
                main_data,
                positions=[0.4],  # Offset to left
                patch_artist=True, 
                showmeans=True,
                meanline=True,
                meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
                medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
                showfliers=True,
                flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
                whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
                capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
                widths=0.4 * scaling_factor
            )
            
            # Color the box
            for box in bp_main['boxes']:
                box.set_facecolor(colors['TUMBlue1'])
                box.set_alpha(0.7)
            
            # Create histogram or points for outliers if they exist
            if len(outlier_data) > 0:
                if len(outlier_data) >= 10:
                    # For many outliers, use histogram
                    outlier_ax.hist(outlier_data, bins=5, color='salmon', alpha=0.8, orientation='vertical')
                else:
                    # For few outliers, use scatter points with small jitter for clarity
                    y_pos = np.ones(len(outlier_data)) * 0.5
                    # Add small vertical jitter if multiple points
                    if len(outlier_data) > 1:
                        y_pos += np.random.normal(0, 0.1, size=len(outlier_data))
                    outlier_ax.scatter(outlier_data, y_pos, color='salmon', marker='o', s=30, alpha=0.8)
                    
                # Add count of outliers as title
                outlier_ax.set_title(f'n={len(outlier_data)}', fontsize=10 * scaling_factor)
                
                # Format y-axis for outlier plot
                if len(outlier_data) >= 10:
                    # Show counts for histogram
                    outlier_ax.set_ylabel('Count', fontsize=8 * scaling_factor)
                else:
                    # Hide y-axis for scatter
                    outlier_ax.set_yticks([])
                
                # Format x-axis as percentage
                outlier_ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
                
                # Set limits for outlier plot - stretch a bit beyond the data for visual clarity
                outlier_min = min(outlier_data.min() * 1.1, -0.5)  # Don't go below -50%
                outlier_ax.set_xlim([outlier_min, cutoff])
            else:
                # If no outliers, show a message
                outlier_ax.text(0.5, 0.5, 'No outliers\n< -20%', 
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=outlier_ax.transAxes,
                               fontsize=8 * scaling_factor,
                               fontstyle='italic',
                               color='gray')
                outlier_ax.set_yticks([])
                outlier_ax.set_xticks([])
            
            # Define legend handles
            mean_line = mlines.Line2D([], [], color=colors['TUMGreen3'], linestyle='--', label='Mean', linewidth=2 * scaling_factor)
            median_line = mlines.Line2D([], [], color=colors['TUMOrange'], label='Median', linewidth=2 * scaling_factor)
            
            # Add legend to main plot
            main_ax.legend(handles=[median_line, mean_line], loc='upper right', fontsize=10 * scaling_factor, frameon=True)
            
            # Annotate median value on main plot
            median_value = main_data.median()
            median_feature = bp_main['medians'][0]
            x_median, y_median = median_feature.get_xydata()[1]
            main_ax.text(0.6 * scaling_factor, y_median, 
                         f'{median_value:.2%}', 
                         horizontalalignment='center',
                         color=colors['TUMBlack'], 
                         fontsize=10 * scaling_factor, 
                         bbox=dict(boxstyle='round', pad=0.2 * scaling_factor, 
                                  facecolor=colors['TUMWhite'], 
                                  edgecolor=colors['TUMOrange']))
            
            # Add grid lines to main plot
            main_ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
            
            # Set title for main plot
            main_ax.set_title(f'Freight Forwarder {ff}', fontsize=14 * scaling_factor)
            
            # Define y-axis limits and format as percentage
            main_ax_min = cutoff
            main_ax_max = max(main_data.max() * 1.1, 1.0)
            main_ax.set_ylim([main_ax_min, main_ax_max])
            main_ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            main_ax.set_ylabel('State of Charge (SoC)', fontsize=12 * scaling_factor)
            
            # Hide x-axis labels and ticks for main plot since it's just one boxplot
            main_ax.set_xticks([])
            main_ax.tick_params(axis='x', which='both', length=0)
            
            # Add horizontal line at 0% for negative values
            main_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            
            # Add a dashed line at cutoff value
            main_ax.axhline(y=cutoff, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            
            # Connect the main and outlier plots with a horizontal line
            # This creates a visual indicator that the plots are related
            connection = plt.Line2D([0.76, 0.79], [0.5, 0.5], 
                                  transform=fig.transFigure, 
                                  color='red', 
                                  linestyle='--',
                                  linewidth=0.8,
                                  alpha=0.7)
            fig.lines.append(connection)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save each figure separately
        plt.savefig(f'data/output/figures/energy/freight_forwarder_min_soc_boxplots_{fig_num+1}.svg', bbox_inches='tight')
        plt.savefig(f'data/output/figures/energy/freight_forwarder_min_soc_boxplots_{fig_num+1}.pdf', bbox_inches='tight')
        plt.show()
    

# -------------------------------------------------------------------------------------------
#                           SOC BY FREIGHT FORWARDER BOXPLOTS (ENERGY)
# -------------------------------------------------------------------------------------------


def plot_soc_by_freight_forwarder(df_tours_energy, charging_powers):
    """
    Create boxplots showing the distribution of minimum, SoC at departure, and SoC at arrival values
    for each freight forwarder, with each freight forwarder in a separate plot.
    Plot is cut off at -20% SoC but boxplot statistics include all data.
    
    Parameters:
    -----------
    df_tours_energy : pandas.DataFrame
        DataFrame containing tour energy data with soc_min, soc_at_departure, soc_at_arrival, and freight_forwarder columns
    """
    df = df_tours_energy.copy()
    charging_power = charging_powers['home base']
    
    # Convert freight_forwarder to string for consistent plotting
    df['freight_forwarder'] = df['freight_forwarder'].astype(str)
    
    # Get unique freight forwarders
    freight_forwarders = sorted(df['freight_forwarder'].unique())
    
    # Define plot size
    textwidth = (159.2 / 25.4)*2.2  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    height = h_169 * 1.8  # Increase height for better readability
    width = textwidth * 1.2  # Increase width for better layout
    
    # Define cutoff for visualization
    cutoff = -0.5  # -50% SoC
    
    # Create single figure with 3x2 grid layout like plot_weekly_energy_demand_boxplot2
    fig = plt.figure(figsize=(width, height))
    
    # Add a GridSpec with 3 rows and 2 columns
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
    # Create the individual freight forwarder axes
    axes = []
    for i in range(3):  # Rows 0-2
        for j in range(2):  # 2 columns
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Process each freight forwarder
    for i, ff in enumerate(freight_forwarders):
        if i >= len(axes):
            break
            
        # Filter data for current freight forwarder
        ff_data = df[df['freight_forwarder'] == ff]
        
        # Calculate and print SoC statistics for the current freight forwarder
        print(f"\nFreight Forwarder {ff} - SoC Statistics:")
        for soc_type in ['soc_min', 'soc_at_departure', 'soc_at_arrival']:
            data = ff_data[soc_type]
            print(f"{soc_type}: Mean={data.mean():.2%}, Median={data.median():.2%}, Min={data.min():.2%}, Max={data.max():.2%}")
            # Count outliers below cutoff
            outliers = data[data < cutoff]
            if len(outliers) > 0:
                print(f"  {soc_type} outliers below {cutoff:.0%}: {len(outliers)} (minimum: {outliers.min():.2%})")
        
        # Prepare boxplot data
        min_soc_data = ff_data['soc_min']
        departure_soc_data = ff_data['soc_at_departure'] 
        arrival_soc_data = ff_data['soc_at_arrival']
        
        # Create boxplots for min, departure, and arrival SoC
        scaling_factor = textwidth / 16
        
        # Set positions for box plots
        positions = [0.6, 1.0, 1.4]
        labels = ['Min', 'At Departure', 'At Arrival']
        
        # Create boxplots
        bp_min = axes[i].boxplot(
            min_soc_data, 
            positions=[positions[0]], 
            patch_artist=True, 
            showmeans=True,
            meanline=True,
            meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
            medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
            showfliers=True,
            flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
            whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            widths=0.25 * scaling_factor
        )
        
        bp_departure = axes[i].boxplot(
            departure_soc_data, 
            positions=[positions[1]], 
            patch_artist=True, 
            showmeans=True,
            meanline=True,
            meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
            medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
            showfliers=True,
            flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
            whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            widths=0.25 * scaling_factor
        )
        
        bp_arrival = axes[i].boxplot(
            arrival_soc_data, 
            positions=[positions[2]], 
            patch_artist=True, 
            showmeans=True,
            meanline=True,
            meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
            medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
            showfliers=True,
            flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
            whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            widths=0.25 * scaling_factor
        )
        
        # Color boxes
        for box in bp_min['boxes']:
            box.set_facecolor(colors['TUMBlue1'])
            box.set_alpha(0.7)
        
        for box in bp_departure['boxes']:
            box.set_facecolor(colors['TUMBlue2'])
            box.set_alpha(0.7)
            
        for box in bp_arrival['boxes']:
            box.set_facecolor(colors['TUMBlue3'])
            box.set_alpha(0.7)
        
        # Set x-axis and y-axis tick labels font size 
        axes[i].tick_params(axis='x', labelsize=12 * scaling_factor)
        axes[i].tick_params(axis='y', labelsize=12 * scaling_factor)
        
        # Define legend handle 
        mean_line = mlines.Line2D([], [], color=colors['TUMGreen3'], linestyle='--', label='Mean', linewidth=3 * scaling_factor)
        median_line = mlines.Line2D([], [], color=colors['TUMOrange'], label='Median', linewidth=3 * scaling_factor)
        
        # Add legend
        axes[i].legend(handles=[median_line, mean_line], loc='upper right', fontsize=12 * scaling_factor, frameon=True)
        
        # Annotate median values next to the boxes 
        for pos, bp, data in zip(positions, [bp_min, bp_departure, bp_arrival], [min_soc_data, departure_soc_data, arrival_soc_data]):
            median_value = data.median()
            median_feature = bp['medians'][0]
            x_median, y_median = median_feature.get_xydata()[1]
            axes[i].text(pos + 0.15 * scaling_factor, y_median, 
                         f'{median_value:.2%}', 
                         horizontalalignment='center',
                         color=colors['TUMBlack'], 
                         fontsize=12 * scaling_factor,  
                         bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, 
                                  facecolor=colors['TUMWhite'], 
                                  edgecolor=colors['TUMOrange']))
        
        # Set x-axis tick positions and labels
        axes[i].set_xticks(positions)
        axes[i].set_xticklabels(labels)
        
        # Add grid lines
        axes[i].grid(True, axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
        
        # Set title for each subplot
        axes[i].set_title(f'Freight Forwarder {ff} - at {charging_power} kW Depot Charging', fontsize=16 * scaling_factor)
        
        # Set y-axis limits with cutoff at -20%
        # Determine max value based on data range
        max_value = max(min_soc_data.max(), departure_soc_data.max(), arrival_soc_data.max())
        y_max = max(max_value * 1.1, 1.0)  # Always include 100% in the plot
        
        axes[i].set_ylim([cutoff, y_max])
        axes[i].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        
        # Adjust tick spacing based on range
        y_range = y_max - cutoff
        major_tick = 0.2 if y_range > 1.0 else 0.1
        minor_tick = major_tick / 2
        
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(major_tick))
        axes[i].yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick))
        
        # Add a horizontal line at 0% for reference
        axes[i].axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add a horizontal line at cutoff to show where data is cut off
        axes[i].axhline(y=cutoff, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add note about outliers below cutoff
        outlier_notes = []
        for soc_type, data in [('Min', min_soc_data), ('Departure', departure_soc_data), ('Arrival', arrival_soc_data)]:
            outliers = data[data < cutoff]
            if len(outliers) > 0:
                outlier_notes.append(f"{soc_type}: {len(outliers)} outliers, lowest SoC: {outliers.min():.0%} \%")
        
        if outlier_notes:
            note_text = f"Outliers below -{int(cutoff*100)} \%:\n" + "\n".join(outlier_notes)
            axes[i].text(0.02, 0.02, note_text, 
                        transform=axes[i].transAxes,
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        fontsize=12 * scaling_factor, 
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='white',  # Changed to white background
                                 edgecolor='gray'))
        
        # Customize spines
        for spine in axes[i].spines.values():
            spine.set_edgecolor(colors['TUMBlack'])
            spine.set_linewidth(0.8 * scaling_factor)
        
        # Add individual axis labels to each subplot
        #axes[i].set_xlabel('SoC Type', fontsize=14 * scaling_factor)
        #axes[i].set_ylabel('State of Charge (SoC)', fontsize=14 * scaling_factor)
    
    # Remove any empty subplots if there are fewer than 6 freight forwarders
    for i in range(len(freight_forwarders), len(axes)):
        fig.delaxes(axes[i])
    
    # Add common labels for the figure
    fig.text(0.5, 0.01, 'SoC Type', ha='center', va='center', fontsize=16 * scaling_factor)
    fig.text(0.01, 0.5, 'State of Charge (SoC)', ha='center', va='center', rotation='vertical', fontsize=16 * scaling_factor)
    
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])  # Adjust layout to make room for common labels
    
    # Save figure
    plt.savefig(f'data/output/figures/energy/freight_forwarder_soc_boxplots_{charging_power}kW.svg', bbox_inches='tight')
    plt.savefig(f'data/output/figures/energy/freight_forwarder_soc_boxplots_{charging_power}kW.pdf', bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------------
#                              REST TIME KDEs BY LOCATION
# ------------------------------------------------------------------------------


def plot_rest_time_kde(df_stops):
    """
    Plot KDEs for rest times segmented by location.
    Uses the same colors for locations as in plot_fleet_occupation() for consistency.
    """
    # Define exact color mapping to match the screenshot legend
    location_colors = {
        'industrial area': colors_plot[2] ,   # blue
        'rest area': colors_plot[4],         # purple/pink
        'home base': colors_plot[1],         # orange
        'other area': colors_plot[3],        # green
        'service area fuel': colors_plot[5]  # yellow
    }

    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_43 = 6  # Define plot height
    cut_time_1 = 2  # First cut-off time for rest in hours
    cut_time_2 = 6  # Second cut-off time for rest in hours
    cut_time_max = 20  # Maximum time for rest in hours
    indicator_level_main = 2.5  # Main indicator level for highlighting
    indicator_level_sub_1 = 2.5  # Sub-indicator level for short rests
    indicator_level_sub_2 = 0.4  # Sub-indicator level for long rests
    indicator_color = '#202020'  # Color for indicator lines

    gf_loc = GridFigure(1, 1, textwidth, h_43 / 2, wspace=0.2, hspace=0.3, legend_mode='above', constrained_layout=False)
    gf_loc_2 = GridFigure(1, 2, textwidth, h_43 / 2, wspace=0.2, constrained_layout=False)
    gf_loc_axes = np.atleast_1d(gf_loc.axes_list).flatten()
    gf_loc_2_axes = np.atleast_1d(gf_loc_2.axes_list).flatten()

    plot_data = df_stops.loc[(df_stops.rest_time <= 24 * 3600)]
    plot_data_2h = df_stops.loc[(df_stops.rest_time <= cut_time_1 * 3600)]
    plot_data_24h = df_stops.loc[(df_stops.rest_time > cut_time_2 * 3600) & (df_stops.rest_time <= 24 * 3600)]

    for data in [plot_data, plot_data_2h, plot_data_24h]:
        if 'rest_time_h' not in data.columns or 'location' not in data.columns:
            raise ValueError("Missing required columns in the data")
        if data[['rest_time_h', 'location']].isnull().any().any():
            raise ValueError("NaNs found in required columns")

    for data, bw_adjust, ax in zip([plot_data, plot_data_2h, plot_data_24h], [.002, .2, .2], [gf_loc_axes[0], *gf_loc_2_axes]):
        # Use the exact color palette from the screenshot
        sns.kdeplot(data=data, x='rest_time_h', hue='location', bw_adjust=bw_adjust, palette=location_colors, cut=0, common_norm=False, ax=ax)

    gf_loc_axes[0].plot([0, cut_time_1], [indicator_level_main] * 2, "--v", linewidth=1, markersize=5, color=indicator_color)
    gf_loc_axes[0].plot([cut_time_2, cut_time_max], [indicator_level_main] * 2, "--o", linewidth=1, markersize=5, color=indicator_color)
    gf_loc_2_axes[0].plot([0, cut_time_1], [indicator_level_sub_1] * 2, "--v", linewidth=1, markersize=5, color=indicator_color)
    gf_loc_2_axes[1].plot([cut_time_2, cut_time_max], [indicator_level_sub_2] * 2, "--o", linewidth=1, markersize=5, color=indicator_color)

    gf_loc.legend_ax.axis("off")

    for ax, major_x_grid, minor_x_grid, major_y_grid, minor_y_grid, xlim, ylim, xlabel, ylabel in zip(
            [gf_loc_axes[0], *gf_loc_2_axes], [4, 0.5, 2], [0.5, 0.1, 1], [0.5, 0.5, 0.1], [0.25, 0.1, 0.05],
            [(0, 24), (0, cut_time_1), (cut_time_2, cut_time_max)], [(0, 3), (0, 3), (0, 0.5)], ['rest time / h'] * 3,
            ['density'] * 2 + ['']):
        ax.grid(which='major', alpha=alpha_major)
        ax.grid(which='minor', alpha=alpha_minor)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_grid))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_grid))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_y_grid))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y_grid))
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.get_legend().remove()
        setFontSize([ax], 9)

    # Create legend with the consistent colors from screenshot
    setFontSize(gf_loc_axes, 9)
    handles = [genLineLegendHandle(location_colors[loc]) for loc in location_colors]
    legend_labels = list(location_colors.keys())
    gf_loc.legend_ax.legend(handles=handles, labels=legend_labels, ncol=3, loc="center")

    gf_loc.fig.savefig(f'data/output/figures/operational/rest_time_global.svg', bbox_inches='tight')
    gf_loc_2.fig.savefig(f'data/output/figures/operational/rest_time_detail.svg', bbox_inches='tight')

    gf_loc.fig.savefig(f'data/output/figures/operational/rest_time_global.pdf', bbox_inches='tight')
    gf_loc_2.fig.savefig(f'data/output/figures/operational/rest_time_detail.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------------
#                             FLEET OCCUPATION PLOT
# ------------------------------------------------------------------------------


def plot_fleet_occupation(df_occ, truck_day):

    # Prepare figure
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    gridfig = GridFigure(1,2,textwidth, h_169, wspace=0.35, hspace=.5, legend_mode="above", constrained_layout=False, ratios={'height':[1], 'width':[1, 2]})
    ax = gridfig.axes_list[0] #len(df_occ['freight_forwarder'].unique())]
    LOCATIONS = dp.LOCATIONS
    
    #colors_fleets = {i: color_list[i - 1] for i in range(1, 5)}
    #colors_locations = {loc: color_list[i + 4] for i, loc in enumerate([*LOCATIONS, 'other_area', 'driving'])}
    #colors_locations_2 = {loc.replace('_', ' '): color_list[i + 4] for i, loc in enumerate([*LOCATIONS, 'other_area', 'driving'])}

    # Plot
    # df_occ: absolute numbers per vehicle in each occupation 
    plot_data = df_occ.drop(columns=['freight_forwarder']).transpose()
    plot_data = (plot_data / plot_data.sum()).transpose() #relative values per vehicle
    plot_data['freight_forwarder']=df_occ['freight_forwarder']
    plot_data = plot_data.fillna(0)
    plot_data = plot_data.groupby('freight_forwarder').agg({
        'driving' : 'mean',
        'home_base' : 'mean',
        'industrial_area' : 'mean',
        'other_area' : 'mean',
        'rest_area' : 'mean',
        'service_area_fuel' : 'mean'
    })
    #plot_data = plot_data[['driving','home_base','industrial_area','other_area','rest_area','service_area_fuel']] #rearange columns
    plot_data.plot.bar(ax = ax, stacked=True, color=colors_plot, width=0.5)

    """
    for fleet in plot_data['freight_forwarder'].unique():
        plot_data_fleet = plot_data[plot_data['freight_forwarder']==fleet]
        plot_data_fleet = plot_data_fleet[['driving','home_base','industrial_area','other_area','rest_area','service_area_fuel']] #rearange columns
        plot_data_fleet = plot_data_fleet.mean().to_frame().T
        plot_data_fleet.plot.bar(ax = ax[fleet - 1], stacked=True, color=colors_fleets, width=0.5)
    """

    #plot_data.plot.bar(ax = ax, stacked=True, color=colors_locations, width=0.5)

    # Adjust figure
    ax.set(
        ylim=(0,1),
        yticks=np.arange(0,1.04,0.1),
        xlabel="fleet",
        ylabel="share of vehicle status by fleet"
    )

    ax.grid(axis="y", which='major', alpha=alpha_major)
    ax.grid(axis="y", which='minor', alpha=alpha_minor)

    # Create legend - Apply percentage formatter more explicitly
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.05))

    handles,labels = ax.get_legend_handles_labels()
    gridfig.legend_ax.axis("off")
    gridfig.legend_ax.legend(handles = handles, labels=[l.replace("_"," ") for l in labels], ncol=3, loc="center")
    ax.get_legend().remove()

    # Adjust fonts
    ax = gridfig.axes_list[1]

    """ -------- 2nd plot -------- """
    # Plot
    plot_data = (truck_day / truck_day.sum()).transpose()
    plot_data.plot.bar(ax=ax, stacked=True, color=colors_plot)

    # Adjust figure
    ax.set(
        ylim=(0,1),
        yticks=np.arange(0,1.04,0.1),
        xlabel="hour",
        ylabel="share of vehicle status by time"
    )

    # Create legend - Apply percentage formatter more explicitly
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t, pos: f"{round(t*100)}%"))
    handles,labels = ax.get_legend_handles_labels()
    gridfig.legend_ax.axis("off")
    gridfig.legend_ax.legend(handles = handles, labels=[l.replace("_"," ") for l in labels], ncol=3, loc="center")
    ax.get_legend().remove()

    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.05))

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.grid(axis="y", which='major', alpha=alpha_major)
    ax.grid(axis="y", which='minor', alpha=alpha_minor)

    for ax_i in gridfig.axes_list:
        for tick in ax_i.get_xticklabels():
            tick.set(rotation=0)#,ha='right')

    # Adjust fonts
    setFontSize(gridfig.axes_list+[gridfig.legend_ax],9)

    # Save figure
    plt.savefig('data/output/figures/operational/fleet_occupation.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/operational/fleet_occupation.pdf', bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------------
#                              LOCATION LOAD PROFILES
# ------------------------------------------------------------------------------


def plot_weekly_energy_demand_boxplot2(df_loads):
    """
    Creates a figure with a central boxplot showing all home bases' energy demand together
    at the top, and individual plots for each home base (CID) below.
    
    Parameters:
    -----------
    df_loads : DataFrame
        DataFrame containing daily energy demand data with day, energy_demand_kwh, cid, and freight_forwarder columns
    """
    
    # Convert day column to datetime and extract weekday
    df_loads['day'] = pd.to_datetime(df_loads['day'])
    df_loads['weekday'] = df_loads['day'].dt.weekday  # 0 = Monday, 6 = Sunday
    
    # Define plot size
    textwidth = (159.2 / 25.4)*2.2  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    height = h_169 * 2.5  # Increase height for better readability
    width = textwidth * 1.2  # Increase width for better layout

    # Define colors for weekdays and average
    weekday_colors = {
        0: colors['TUMBlue4'],  # Monday
        1: colors['TUMBlue3'],  # Tuesday
        2: colors['TUMBlue2'],  # Wednesday 
        3: colors['TUMBlue1'],  # Thursday
        4: colors['TUMGray3'],  # Friday
        5: colors['TUMGray2'],  # Saturday 
        6: colors['TUMGray1'],  # Sunday
        'Combined': colors['LightPurple']  # Average/Combined column
    }

    # Create subplots with a 3 column x 5 row grid (15 subplots total)
    fig = plt.figure(figsize=(width, height))
    
    # Add a GridSpec with 5 rows and 3 columns
    gs = fig.add_gridspec(5, 3, height_ratios=[2, 1, 1, 1, 1])
    
    # Add the central plot at the top, spanning all columns
    central_ax = fig.add_subplot(gs[0, :])
    
    # Create the individual CID axes
    axes = []
    for i in range(1, 5):  # Rows 1-4
        for j in range(3):  # 3 columns
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Get unique CIDs
    cids = sorted(df_loads['cid'].unique())
    
    # Prepare data for the central plot
    all_boxplot_data = []
    all_weekday_labels = []
    all_weekday_numbers = []  # Store the actual weekday numbers
    
    for weekday in range(7):
        weekday_data = df_loads[df_loads['weekday'] == weekday]['energy_demand_kwh'].tolist()
        if len(weekday_data) > 0:  # Only add if there's data for this weekday
            all_boxplot_data.append(weekday_data)
            all_weekday_labels.append(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday])
            all_weekday_numbers.append(weekday)
    
    # Add the average data for the central plot (all weekdays combined)
    all_avg_data = df_loads['energy_demand_kwh'].tolist()
    all_boxplot_data.append(all_avg_data)
    all_weekday_labels.append('Combined')
    all_weekday_numbers.append('Combined')
    
    # Create central boxplot with all data
    scaling_factor = textwidth / 16
    bp_central = central_ax.boxplot(
        all_boxplot_data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
        medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
        showfliers=True,
        flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
        whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
        capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
        widths=0.55 * scaling_factor,
        positions=range(len(all_boxplot_data)),
        labels=all_weekday_labels
    )
    
    # Style central boxplot
    central_ax.set_title('All Home Bases Combined', fontsize=14 * scaling_factor)
    central_ax.tick_params(axis='x', labelsize=12 * scaling_factor)
    central_ax.tick_params(axis='y', labelsize=12 * scaling_factor)
    central_ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
    central_ax.set_ylim(bottom=0)
    
    # Add individual axis labels to central plot
    central_ax.set_xlabel('Weekday', fontsize=12 * scaling_factor)
    central_ax.set_ylabel('Energy Demand (kWh)', fontsize=12 * scaling_factor)
    
    # Define legend handles directly in the central plot
    mean_line = mlines.Line2D([], [], color=colors['TUMGreen3'], linestyle='--', label='Mean', linewidth=3 * scaling_factor)
    median_line = mlines.Line2D([], [], color=colors['TUMOrange'], label='Median', linewidth=3 * scaling_factor)
    # Add the legend to the top left corner of the central plot
    central_ax.legend(handles=[median_line, mean_line], loc='upper left', fontsize=12 * scaling_factor, frameon=True, ncol=2)
    
    # Assign colors to the central boxplots based on weekday or 'Combined'
    for i, (box, weekday_or_combined) in enumerate(zip(bp_central['boxes'], all_weekday_numbers)):
        box.set_facecolor(weekday_colors[weekday_or_combined])
        # Add a slightly thicker border to the average box to make it stand out
        if weekday_or_combined == 'Combined':  # Average box
            box.set(linewidth=2.0 * scaling_factor)
    
    # Annotate central plot median values
    for n, (median_feature, demands) in enumerate(zip(bp_central['medians'], all_boxplot_data)):
        demands_series = pd.Series(demands)
        median_value = demands_series.median()
        x_median, y_median = median_feature.get_xydata()[1]
        central_ax.text(x_median, y_median, f'{median_value:.2f}', 
                horizontalalignment='center', color=colors['TUMBlack'], fontsize=10 * scaling_factor, 
                bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=colors['TUMWhite'], 
                        edgecolor=colors['TUMOrange'], alpha=0.9))
    
    # Process each CID for individual plots
    for i, cid in enumerate(cids):
        if i >= len(axes):
            break
            
        # Filter data for current CID
        cid_data = df_loads[df_loads['cid'] == cid]
        
        # Get the freight forwarder for this CID
        ff = cid_data['freight_forwarder'].iloc[0] if not cid_data.empty else "Unknown"
        
        # Calculate and print daily energy demand per home base
        print(f"\nHome Base CID {cid} - Daily Energy Demand:")
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = cid_data.groupby('weekday')['energy_demand_kwh'].agg(['mean', 'std', 'median', 'count']).reset_index()
        daily_stats['weekday_name'] = daily_stats['weekday'].apply(lambda x: weekday_names[x])
        print(daily_stats[['weekday_name', 'mean', 'std', 'median', 'count']])
        print(f"Overall mean: {cid_data['energy_demand_kwh'].mean():.2f} kWh")
        
        # Prepare boxplot data with only available weekdays
        boxplot_data = []
        weekday_labels = []
        weekday_numbers = []  # Store the actual weekday numbers
        
        for weekday in range(7):
            weekday_data = cid_data[cid_data['weekday'] == weekday]['energy_demand_kwh'].tolist()
            if len(weekday_data) > 0:  # Only add if there's data for this weekday
                boxplot_data.append(weekday_data)
                weekday_labels.append(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday])
                weekday_numbers.append(weekday)
        
        # Add the average data for this CID (all weekdays combined)
        avg_data = cid_data['energy_demand_kwh'].tolist()
        boxplot_data.append(avg_data)
        weekday_labels.append('Combined')
        weekday_numbers.append('Combined')
        
        # Create boxplot
        bp = axes[i].boxplot(
            boxplot_data,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
            medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
            showfliers=True,
            flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
            whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            widths=0.55 * scaling_factor,
            positions=range(len(boxplot_data)),
            labels=weekday_labels
        )
        
        # Set x-axis and y-axis tick labels font size
        axes[i].tick_params(axis='x', labelsize=10 * scaling_factor)
        axes[i].tick_params(axis='y', labelsize=10 * scaling_factor)
        
        # Add individual axis labels to each subplot
        axes[i].set_xlabel('Weekday', fontsize=12 * scaling_factor)
        axes[i].set_ylabel('Energy Demand (kWh)', fontsize=12 * scaling_factor)
        
        # Annotate median values
        for n, (median_feature, demands) in enumerate(zip(bp['medians'], boxplot_data)):
            demands_series = pd.Series(demands)
            median_value = demands_series.median()
            x_median, y_median = median_feature.get_xydata()[1]
            axes[i].text(x_median, y_median, f'{median_value:.2f}', 
                    horizontalalignment='center', color=colors['TUMBlack'], fontsize=8 * scaling_factor, 
                    bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=colors['TUMWhite'], 
                            edgecolor=colors['TUMOrange'], alpha=0.9))
        
        # Assign colors to the boxplots based on weekday or 'Combined'
        for j, (box, weekday_or_combined) in enumerate(zip(bp['boxes'], weekday_numbers)):
            box.set_facecolor(weekday_colors[weekday_or_combined])
            # Add a slightly thicker border to the average box to make it stand out
            if weekday_or_combined == 'Combined':  # Average box
                box.set(linewidth=2.0 * scaling_factor)
        
        # Set title for each subplot with both freight forwarder and CID
        axes[i].set_title(f'Freight Forwarder {ff} - Home Base CID {cid}', fontsize=12 * scaling_factor)
        # Add legend to each plot in the top right corner
        axes[i].legend(handles=[median_line, mean_line], loc='upper left', fontsize=11 * scaling_factor, frameon=True, ncol=2)
        # Add grid
        axes[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
        
        # Set y-axis limit
        axes[i].set_ylim(bottom=0)
        
        # Customize spines
        for spine in axes[i].spines.values():
            spine.set_edgecolor(colors['TUMBlack'])
            spine.set_linewidth(0.8 * scaling_factor)
    
    # Remove any empty subplots if there are fewer than 12 CIDs
    for i in range(len(cids), len(axes)):
        fig.delaxes(axes[i])
    
    # Still add common labels as a reference, but they'll be less necessary with individual labels
    fig.text(0.5, 0.01, 'Weekday', ha='center', va='center', fontsize=16 * scaling_factor)
    fig.text(0.01, 0.5, 'Energy Demand (kWh)', ha='center', va='center', rotation='vertical', fontsize=16 * scaling_factor)

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])  # Adjust layout to make room for common labels
    plt.savefig('data/output/figures/energy/home_base_weekday_energy_boxplots.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/energy/home_base_weekday_energy_boxplots.pdf', bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------------
#                             FLEET OCCUPATION PLOT
# ------------------------------------------------------------------------------


def plot_weekly_energy_demand_boxplot(df_loads):
    """
    Creates a figure with a central boxplot showing all home bases' energy demand together
    at the top, and individual plots for each home base (CID) below.
    
    Parameters:
    -----------
    df_loads : DataFrame
        DataFrame containing daily energy demand data with day, energy_demand_kwh, cid, and freight_forwarder columns
    """
    
    # Convert day column to datetime and extract weekday
    df_loads['day'] = pd.to_datetime(df_loads['day'])
    df_loads['weekday'] = df_loads['day'].dt.weekday  # 0 = Monday, 6 = Sunday
    
    # Define plot size
    textwidth = (159.2 / 25.4)*2.2  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    height = h_169 * 2  # Increase height for better readability
    width = textwidth * 1.2  # Increase width for better layout

    # Define colors for weekdays and average
    weekday_colors = {
        0: colors['TUMBlue4'],  # Monday
        1: colors['TUMBlue3'],  # Tuesday
        2: colors['TUMBlue2'],  # Wednesday 
        3: colors['TUMBlue1'],  # Thursday
        4: colors['TUMGray3'],  # Friday
        5: colors['TUMGray2'],  # Saturday 
        6: colors['TUMGray1'],  # Sunday
        'Combined': colors['LightPurple']  # Average/Combined column
    }

    # Create subplots with a 3 column x 5 row grid (15 subplots total)
    fig = plt.figure(figsize=(width, height))
    
    # Add a GridSpec with 5 rows and 3 columns
    gs = fig.add_gridspec(5, 3, height_ratios=[2, 1, 1, 1, 1])
    
    # Add the central plot at the top, spanning all columns
    central_ax = fig.add_subplot(gs[0, :])
    
    # Create the individual CID axes
    axes = []
    for i in range(1, 5):  # Rows 1-4
        for j in range(3):  # 3 columns
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Get unique CIDs
    cids = sorted(df_loads['cid'].unique())
    
    # Prepare data for the central plot
    all_boxplot_data = []
    all_weekday_labels = []
    all_weekday_numbers = []  # Store the actual weekday numbers
    
    for weekday in range(7):
        weekday_data = df_loads[df_loads['weekday'] == weekday]['energy_demand_kwh'].tolist()
        if len(weekday_data) > 0:  # Only add if there's data for this weekday
            all_boxplot_data.append(weekday_data)
            all_weekday_labels.append(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday])
            all_weekday_numbers.append(weekday)
    
    # Add the average data for the central plot (all weekdays combined)
    all_avg_data = df_loads['energy_demand_kwh'].tolist()
    all_boxplot_data.append(all_avg_data)
    all_weekday_labels.append('Combined')
    all_weekday_numbers.append('Combined')
    
    # Create central boxplot with all data
    scaling_factor = textwidth / 16
    bp_central = central_ax.boxplot(
        all_boxplot_data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
        medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
        showfliers=True,
        flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
        whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
        capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
        widths=0.55 * scaling_factor,
        positions=range(len(all_boxplot_data)),
        labels=all_weekday_labels
    )
    
    # Style central boxplot
    central_ax.set_title('All Home Bases Combined', fontsize=14 * scaling_factor)
    central_ax.tick_params(axis='x', labelsize=12 * scaling_factor)
    central_ax.tick_params(axis='y', labelsize=12 * scaling_factor)
    central_ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
    central_ax.set_ylim(bottom=0)
    
    # Define legend handles directly in the central plot
    mean_line = mlines.Line2D([], [], color=colors['TUMGreen3'], linestyle='--', label='Mean', linewidth=3 * scaling_factor)
    median_line = mlines.Line2D([], [], color=colors['TUMOrange'], label='Median', linewidth=3 * scaling_factor)
    # Add the legend to the top left corner of the central plot
    central_ax.legend(handles=[median_line, mean_line], loc='upper left', fontsize=12 * scaling_factor, frameon=True, ncol=2)
    
    # Assign colors to the central boxplots based on weekday or 'Combined'
    for i, (box, weekday_or_combined) in enumerate(zip(bp_central['boxes'], all_weekday_numbers)):
        box.set_facecolor(weekday_colors[weekday_or_combined])
        # Add a slightly thicker border to the average box to make it stand out
        if weekday_or_combined == 'Combined':  # Average box
            box.set(linewidth=2.0 * scaling_factor)
    
    # Annotate central plot median values
    for n, (median_feature, demands) in enumerate(zip(bp_central['medians'], all_boxplot_data)):
        demands_series = pd.Series(demands)
        median_value = demands_series.median()
        x_median, y_median = median_feature.get_xydata()[1]
        central_ax.text(x_median, y_median, f'{median_value:.2f}', 
                horizontalalignment='center', color=colors['TUMBlack'], fontsize=10 * scaling_factor, 
                bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=colors['TUMWhite'], 
                        edgecolor=colors['TUMOrange'], alpha=0.9))
    
    # Process each CID for individual plots
    for i, cid in enumerate(cids):
        if i >= len(axes):
            break
            
        # Filter data for current CID
        cid_data = df_loads[df_loads['cid'] == cid]
        
        # Get the freight forwarder for this CID
        ff = cid_data['freight_forwarder'].iloc[0] if not cid_data.empty else "Unknown"
        
        # Calculate and print daily energy demand per home base
        print(f"\nHome Base CID {cid} - Daily Energy Demand:")
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = cid_data.groupby('weekday')['energy_demand_kwh'].agg(['mean', 'std', 'median', 'count']).reset_index()
        daily_stats['weekday_name'] = daily_stats['weekday'].apply(lambda x: weekday_names[x])
        print(daily_stats[['weekday_name', 'mean', 'std', 'median', 'count']])
        print(f"Overall mean: {cid_data['energy_demand_kwh'].mean():.2f} kWh")
        
        # Prepare boxplot data with only available weekdays
        boxplot_data = []
        weekday_labels = []
        weekday_numbers = []  # Store the actual weekday numbers
        
        for weekday in range(7):
            weekday_data = cid_data[cid_data['weekday'] == weekday]['energy_demand_kwh'].tolist()
            if len(weekday_data) > 0:  # Only add if there's data for this weekday
                boxplot_data.append(weekday_data)
                weekday_labels.append(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday])
                weekday_numbers.append(weekday)
        
        # Add the average data for this CID (all weekdays combined)
        avg_data = cid_data['energy_demand_kwh'].tolist()
        boxplot_data.append(avg_data)
        weekday_labels.append('Combined')
        weekday_numbers.append('Combined')
        
        # Create boxplot
        bp = axes[i].boxplot(
            boxplot_data,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops=dict(color=colors['TUMGreen3'], linestyle='--', linewidth=1.5 * scaling_factor),
            medianprops=dict(color=colors['TUMOrange'], linewidth=1.5 * scaling_factor),
            showfliers=True,
            flierprops=dict(marker='o', color='red', alpha=0.5, markersize=3),
            whiskerprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            capprops=dict(color=colors['TUMBlack'], linewidth=0.8 * scaling_factor),
            widths=0.55 * scaling_factor,
            positions=range(len(boxplot_data)),
            labels=weekday_labels
        )
        
        # Set x-axis and y-axis tick labels font size
        axes[i].tick_params(axis='x', labelsize=10 * scaling_factor)
        axes[i].tick_params(axis='y', labelsize=10 * scaling_factor)
        
        # Annotate median values
        for n, (median_feature, demands) in enumerate(zip(bp['medians'], boxplot_data)):
            demands_series = pd.Series(demands)
            median_value = demands_series.median()
            x_median, y_median = median_feature.get_xydata()[1]
            axes[i].text(x_median, y_median, f'{median_value:.2f}', 
                    horizontalalignment='center', color=colors['TUMBlack'], fontsize=8 * scaling_factor, 
                    bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=colors['TUMWhite'], 
                            edgecolor=colors['TUMOrange'], alpha=0.9))
        
        # Assign colors to the boxplots based on weekday or 'Combined'
        for j, (box, weekday_or_combined) in enumerate(zip(bp['boxes'], weekday_numbers)):
            box.set_facecolor(weekday_colors[weekday_or_combined])
            # Add a slightly thicker border to the average box to make it stand out
            if weekday_or_combined == 'Combined':  # Average box
                box.set(linewidth=2.0 * scaling_factor)
        
        # Set title for each subplot with both freight forwarder and CID
        axes[i].set_title(f'Freight Forwarder {ff} - Home Base CID {cid}', fontsize=12 * scaling_factor)
        
        # Add grid
        axes[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
        
        # Set y-axis limit
        axes[i].set_ylim(bottom=0)
        
        # Customize spines
        for spine in axes[i].spines.values():
            spine.set_edgecolor(colors['TUMBlack'])
            spine.set_linewidth(0.8 * scaling_factor)
    
    # Remove any empty subplots if there are fewer than 14 CIDs
    for i in range(len(cids), len(axes)):
        fig.delaxes(axes[i])
    
    # Add common labels
    fig.text(0.5, 0.01, 'Weekday', ha='center', va='center', fontsize=16 * scaling_factor)
    fig.text(0.01, 0.5, 'Energy Demand / kWh', ha='center', va='center', rotation='vertical', fontsize=16 * scaling_factor)

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])  # Adjust layout to make room for common labels
    plt.savefig('data/output/figures/energy/home_base_weekday_energy_boxplots.svg', bbox_inches='tight')
    plt.savefig('data/output/figures/energy/home_base_weekday_energy_boxplots.pdf', bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------------
#                             LOAD PROFILES GRID
# ------------------------------------------------------------------------------


def plot_load_profiles_grid(load_profiles, charging_powers, charging_durations):
    """
    Creates a grid of load profiles plots, one for each CID.
    
    Parameters:
    -----------
    load_profiles : dict
        Dictionary of load profiles DataFrames, with CIDs as keys
    charging_powers : dict
        Dictionary of charging powers for each location type
    charging_durations : dict
        Dictionary of average charging durations for each CID
        
    Returns:
    --------
    stats : dict
        Dictionary of statistics for each CID
    """
    tum_colors = {
        'mean': colors['TUMBlue1'],       # Average load
        'max': colors['TUMOrange'],       # Maximum load
        'threshold': colors['TUMGreen3'], # 630 kW threshold
        'grid_major': colors['TUMGray2'], # Major grid
        'grid_minor': colors['TUMGray3'], # Minor grid
    }

    # Create figure with subplots (7 rows, 2 columns)
    fig, axes = plt.subplots(7, 2, figsize=(12, 24))
    axes_flat = axes.flatten()  # Flatten the 2D array for easier indexing
    
    # Store statistics for return
    all_stats = {}
    
    # Loop through each load profile
    for i, (cid, df) in enumerate(load_profiles.items()):
        # Get the appropriate axis for this CID
        ax = axes_flat[i]
        
        charging_power = charging_powers['home base']
        
        # Filter out days where load_kW is all 0
        daily_load_sum = df.groupby(df['date'])['load_kW'].sum()
        active_days = daily_load_sum[daily_load_sum > 0].index.tolist()
        
        # Filter the dataframe to only include active days
        df_active = df[df['date'].isin(active_days)]
        
        # Calculate average and maximum load for each hour
        hourly_avg = df_active.groupby('time')['load_kW'].mean()
        hourly_max = df_active.groupby('time')['load_kW'].max()
        
        # Convert time objects to hour numbers for plotting
        hours = [t.hour + t.minute/60 for t in hourly_avg.index]
        
        # Plot average load
        ax.plot(hours, hourly_avg.values, label='Avg', marker='.', markersize=4, linewidth=1.5, color=tum_colors['mean'])
        # Plot maximum load
        ax.plot(hours, hourly_max.values, label='Max', marker='.', markersize=4, linewidth=1.5, color=tum_colors['max'], linestyle='--')
        
        # Add horizontal line at 630 kW
        #ax.axhline(y=630, color=tum_colors['threshold'], linestyle='-', linewidth=1, label='630 kW')
        
        # Add grid with major and minor lines
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, color=tum_colors['grid_major'], alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color=tum_colors['grid_minor'], alpha=0.5)
        
        # Set title for each subplot with both freight forwarder and CID
        ax.set_title(f'Freight Forwarder {df["freight_forwarder"].iloc[0]} - Base {cid} - Avg Charging Duration: {charging_durations[cid]} min', fontsize=10)
        
        # Add individual x and y axis labels to each subplot
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Load (kW)', fontsize=10)
        
        # Set x-ticks for all plots and make them always visible
        ax.set_xticks(range(0, 24, 4))
        ax.set_xlim(0, 24)
        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Add minor ticks for hours
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        # Add legend only to the first plot
        # if i == 0:
        #     ax.legend(fontsize=8, loc='upper right')
        # Add legend to each plot in the top right corner
        ax.legend(fontsize=10, loc='upper right', framealpha=0.8)
        
        # Calculate and store statistics
        all_stats[cid] = {
            'active_days': len(active_days),
            'avg_max_load': hourly_max.mean(),
            'peak_time': hourly_avg.idxmax() if not hourly_avg.empty else None,
            'peak_avg_load': hourly_avg.max() if not hourly_avg.empty else 0,
        }
    
    # Hide any unused subplots
    for i in range(len(load_profiles), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add a main title for the entire figure
    plt.suptitle(f'Daily Load Profiles for All Home Bases at {charging_power} kW', fontsize=16, fontweight='bold', y=0.95)
    
    # You can still keep the common y-axis label if you want
    # fig.text(0.01, 0.5, 'Load (kW)', ha='center', va='center', rotation='vertical', fontsize=12)
    
    # Adjust spacing between subplots to make room for the individual labels
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])  # Adjust the rect parameters as needed
    
    # Save the figure
    plt.savefig(f'data/output/figures/load_profiles/all_bases_load_{charging_power}kW-profiles.svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'data/output/figures/load_profiles/all_bases_load_{charging_power}kW-profiles.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    return all_stats


# ------------------------------------------------------------------------------
#                             FLEET OCCUPATION PLOT
# ------------------------------------------------------------------------------


def plot_load_profiles_grid_thesis(load_profiles, charging_powers, charging_durations):
    """
    Creates a single figure with 14 subfigures (7 rows × 2 columns), one for each CID.
    
    Parameters:
    -----------
    load_profiles : dict
        Dictionary mapping CID to dataframe containing load profile data
    charging_powers : dict
        Dictionary mapping location types to charging power values
    charging_durations : dict
        Dictionary mapping CID to charging durations
        
    Returns:
    --------
    dict
        Dictionary containing statistics for each CID
    """
    print('Using thesis version of plot_load_profiles_grid')
    tum_colors = {
        'mean': colors['TUMBlue1'],       # Average load
        'max': colors['TUMOrange'],       # Maximum load
        'threshold': colors['TUMGreen3'], # 630 kW threshold
        'threshold2': colors['TUMGreen2'], # 1260 kW threshold
    }

    # Create figure with subplots (7 rows, 2 columns)
    fig, axes = plt.subplots(7, 2, figsize=(12, 24), sharex=True, sharey=False)
    axes = axes.flatten()  # Flatten the 2D array for easier indexing
    
    # Store statistics for return
    all_stats = {}
    
    # Track if we need threshold lines for legend
    has_630_threshold = False
    has_1260_threshold = False
    
    # Loop through each load profile
    for i, (cid, df) in enumerate(load_profiles.items()):
        # Get the appropriate axis for this CID
        ax = axes[i]
        
        charging_power = charging_powers['home base']
        
        # Filter out days where load_kW is all 0
        daily_load_sum = df.groupby(df['date'])['load_kW'].sum()
        active_days = daily_load_sum[daily_load_sum > 0].index.tolist()
        
        # Filter the dataframe to only include active days
        df_active = df[df['date'].isin(active_days)]
        
        # Calculate average and maximum load for each hour
        hourly_avg = df_active.groupby('time')['load_kW'].mean()
        hourly_max = df_active.groupby('time')['load_kW'].max()
        
        # Convert time objects to hour numbers for plotting
        hours = [t.hour + t.minute/60 for t in hourly_avg.index]
        
        # Plot average load
        ax.plot(hours, hourly_avg.values, label='Avg', marker='.', markersize=4, linewidth=1.5, color=tum_colors['mean'])
        # Plot maximum load
        ax.plot(hours, hourly_max.values, label='Max', marker='.', markersize=4, linewidth=1.5, color=tum_colors['max'], linestyle='--')
        
        # Only add threshold lines if data reaches or exceeds the threshold
        max_load_value = max(hourly_max.max() if not hourly_max.empty else 0, hourly_avg.max() if not hourly_avg.empty else 0)
        
        if max_load_value >= 630:
            ax.axhline(y=630, color=tum_colors['threshold'], linestyle='-', linewidth=1, label='630 kW')
            has_630_threshold = True
            
        if max_load_value >= 1260:
            ax.axhline(y=1260, color=tum_colors['threshold2'], linestyle='--', linewidth=1, label='1260 kW')
            has_1260_threshold = True
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add y-axis gridlines every 100 kW
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))  # Optional: minor gridlines every 50 kW
        
        # Set title for each subplot
        ax.set_title(f'Freight Forwarder {df["freight_forwarder"].iloc[0]} - Base {cid} - Avg Charging Duration: {charging_durations[cid]} min', fontsize=10)
        
        # Set x-ticks for all plots
        ax.set_xticks(range(0, 24, 6))
        ax.set_xlim(0, 23)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Add minor ticks for hours
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        # Calculate and store statistics
        all_stats[cid] = {
            'active_days': len(active_days),
            'avg_max_load': hourly_max.mean(),
            'peak_time': hourly_avg.idxmax() if not hourly_avg.empty else None,
            'peak_avg_load': hourly_avg.max() if not hourly_avg.empty else 0,
        }
    
    # Create legend handles based on what's actually used
    legend_handles = []
    legend_labels = []
    
    # Add avg and max to legend
    legend_handles.extend([
        plt.Line2D([0], [0], color=tum_colors['mean'], marker='.', markersize=8, linewidth=2, label='Avg'),
        plt.Line2D([0], [0], color=tum_colors['max'], linestyle='--', marker='.', markersize=8, linewidth=2, label='Max')
    ])
    legend_labels.extend(['Avg', 'Max'])
    
    # Add threshold lines to legend only if they were used
    if has_630_threshold:
        legend_handles.append(plt.Line2D([0], [0], color=tum_colors['threshold'], linestyle='-', linewidth=2))
        legend_labels.append('630 kW')
        
    if has_1260_threshold:
        legend_handles.append(plt.Line2D([0], [0], color=tum_colors['threshold2'], linestyle='--', linewidth=2))
        legend_labels.append('1260 kW')
    
    # Add main title and legend
    plt.suptitle(f'Daily Load Profiles for All Home Bases at {charging_power} kW', fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend below title, arranged horizontally
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.968), 
               ncol=len(legend_handles), fontsize=12, frameon=True)
    
    # Add common axis labels
    fig.text(0.5, 0.02, 'Hour of Day', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Load (kW)', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    
    # Adjust spacing between subplots
    plt.tight_layout(rect=[0.04, 0.03, 0.96, 0.96])
    
    # Save the figure
    plt.savefig(f'data/output/figures/load_profiles/all_bases_load_{charging_power}kW-profiles.svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'data/output/figures/load_profiles/all_bases_load_{charging_power}kW-profiles.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    return all_stats


# ------------------------------------------------------------------------------
#                             LOAD PROFILE PLOT
# ------------------------------------------------------------------------------


def plot_load_profile(df, cid, charging_powers, charging_duration):
    
    # TODO adjust to work with variable charging powers (or just do manually)
    charging_power = charging_powers['home base']
    # Filter out days where load_kW is all 0
    # Group by date and check if the sum of load_kW for that day is 0
    daily_load_sum = df.groupby(df['date'])['load_kW'].sum()
    active_days = daily_load_sum[daily_load_sum > 0].index.tolist()
    
    # Filter the dataframe to only include active days
    df_active = df[df['date'].isin(active_days)]
    
    # Calculate average and maximum load for each hour
    hourly_avg = df_active.groupby('time')['load_kW'].mean()
    hourly_max = df_active.groupby('time')['load_kW'].max()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Convert time objects to hour numbers for plotting
    hours = [t.hour + t.minute/60 for t in hourly_avg.index]
    
    # Plot average load
    plt.plot(hours, hourly_avg.values, label='Average Load', marker='o', linewidth=2, color='blue')
    # Plot maximum load
    plt.plot(hours, hourly_max.values, label='Maximum Load', marker='s', linewidth=2, color='red', linestyle='--')
    
    # Add horizontal line at 630 kW
    plt.axhline(y=630, color='green', linestyle='-', linewidth=2, label='630 kW Threshold')
    
    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Load (kW)', fontsize=12)
    plt.title(f'Daily Load Profile for Base {cid} with {charging_power} kW Charging (Active Days Only)', fontsize=14, fontweight='bold')
    plt.xticks(range(0, 24))
    
    # Add legend
    plt.legend(fontsize=12)
    # Add legend
    plt.legend(fontsize=12)
    
    # Add a text box showing the charging duration
    plt.text(0.5, 0.9, f'Charging Duration: {charging_duration} minutes', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='yellow', alpha=0.5),
             ha='center', fontsize=12)
    
    
    # Show the plot
    plt.tight_layout()

    plt.savefig(f'data/output/figures/load_profiles/cid-{cid}_{charging_power}kW_profile.svg', bbox_inches='tight')
    plt.savefig(f'data/output/figures/load_profiles/cid-{cid}_{charging_power}kW_profile.pdf', bbox_inches='tight')
    plt.show()

    


    # Return summary statistics
    return {
        'active_days': len(active_days),
        'avg_max_load': hourly_max.mean(),
        'peak_time': hourly_avg.idxmax(),
        'peak_avg_load': hourly_avg.max(),
    }


