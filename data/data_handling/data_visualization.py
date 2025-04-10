import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
from fpdf import FPDF
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots

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
    'TUMOrange': '#E37222',
    'TUMBlue2': '#0065BD',
    'TUMGreen3': '#A2AD00',
    'TUMBlue3': '#64A0C8',
    'TUMGreen2': '#515600',
    'TUMBlue4': '#98C6EA',
    'TUMGray1': '#333333',
    'TUMBlue5': '#B0D0F0',
    'TUMGray2': '#7F7F7F',
    'TUMBlue6': '#2A4D70',
    'TUMGray3': '#CCCCCC',
    'TUMBlueDark': '#0B1340',
    'TUMIvory': '#DAD7CB',
    'TUMGreen1': '#292B00'
}

# Remove white and light gray colors from the palette
colors = {k: v for k, v in colors.items() if v not in ['#FFFFFF', '#CCCCCC']}

# Function to calculate brightness
def brightness(hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    return hls[1]  # Luminance value

# Sort color_list by brightness
color_list = list(colors.values())
color_list_sorted = sorted(color_list, key=brightness, reverse=True)

cmap = ListedColormap(color_list_sorted[:97])


# ------------------------------------------------------------------------------
#                              TEMPORAL PATTERN
# ------------------------------------------------------------------------------


def plot_kde_plots(df_trips):
    """
    Plot four KDE plots in a 2x2 grid to show the distribution of the data with vehicle IDs coloring.
    """
    textwidth = 159.2 / 25.4  # Umrechnung der Textbreite von mm in Zoll
    height = textwidth * 1.2  # Etwas höhere Höhe
    width = textwidth * 1.5  # Größere Breite
    
    fig, axes = plt.subplots(2, 2, figsize=(width, height), sharey=False)

    # Define custom color palette stretched over the number of unique vehicle_ids
    n_vehicles = df_trips['vehicle_id'].nunique()
    norm = Normalize(vmin=1, vmax=n_vehicles)
    vehicle_ids = df_trips['vehicle_id'].unique()
    palette = {vehicle_id: cmap(norm(i)) for i, vehicle_id in enumerate(vehicle_ids)}

    sns.kdeplot(data=df_trips, x='distance_km', hue='vehicle_id', ax=axes[0, 0], palette=palette, common_norm=False, fill=True)
    axes[0, 0].set_xlabel('Distance / km')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend_.remove()
    axes[0, 0].set_xlim(0, 250)
    mean_distance = df_trips['distance_km'].mean()
    std_distance = df_trips['distance_km'].std()
    axes[0, 0].axvline(mean_distance, color='r', linestyle='--')
    axes[0, 0].axvline(mean_distance + std_distance, color='r', linestyle=':')
    axes[0, 0].axvline(mean_distance - std_distance, color='r', linestyle=':')

    sns.kdeplot(data=df_trips, x='duration_h', hue='vehicle_id', ax=axes[0, 1], palette=palette, common_norm=False, fill=True)
    axes[0, 1].set_xlabel('Duration / h')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend_.remove()
    axes[0, 1].set_xlim(0, 10)
    mean_duration = df_trips['duration_h'].mean()
    std_duration = df_trips['duration_h'].std()
    axes[0, 1].axvline(mean_duration, color='r', linestyle='--')
    axes[0, 1].axvline(mean_duration + std_duration, color='r', linestyle=':')
    axes[0, 1].axvline(mean_duration - std_duration, color='r', linestyle=':')

    sns.kdeplot(data=df_trips, x='max_speed_kmh', hue='vehicle_id', ax=axes[1, 0], palette=palette, common_norm=False, fill=True)
    axes[1, 0].set_xlabel('Max Speed / km/h')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend_.remove()
    axes[1, 0].set_xlim(0, 150)
    mean_max_speed = df_trips['max_speed_kmh'].mean()
    std_max_speed = df_trips['max_speed_kmh'].std()
    axes[1, 0].axvline(mean_max_speed, color='r', linestyle='--')
    axes[1, 0].axvline(mean_max_speed + std_max_speed, color='r', linestyle=':')
    axes[1, 0].axvline(mean_max_speed - std_max_speed, color='r', linestyle=':')

    sns.kdeplot(data=df_trips, x='avg_speed_kmh', hue='vehicle_id', ax=axes[1, 1], palette=palette, common_norm=False, fill=True)
    axes[1, 1].set_xlabel('Average Speed / km/h')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend_.remove()
    axes[1, 1].set_xlim(0, 100)
    mean_avg_speed = df_trips['avg_speed_kmh'].mean()
    std_avg_speed = df_trips['avg_speed_kmh'].std()
    axes[1, 1].axvline(mean_avg_speed, color='r', linestyle='--')
    axes[1, 1].axvline(mean_avg_speed + std_avg_speed, color='r', linestyle=':')
    axes[1, 1].axvline(mean_avg_speed - std_avg_speed, color='r', linestyle=':')

    # Adjust layout to make room for the single legend on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Add a single color bar on the right
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, aspect=30, shrink=0.8)
    cbar.set_label('Vehicle IDs', labelpad=15, fontsize=12)

    # Add a legend for the red lines inside the top right subplot
    red_lines = [plt.Line2D([0], [0], color='r', linestyle='--', label='Mean'),
                 plt.Line2D([0], [0], color='r', linestyle=':', label='1 Std. Dev.')]
    axes[0, 1].legend(handles=red_lines, loc='upper right', fontsize=8, frameon=True)
    
    plt.savefig('data/output/figures/temporal_pattern/kde_plots.svg', bbox_inches='tight')
    plt.show()


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
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['weekday'] = df_trips['start_time'].dt.weekday  # 0 = Monday, 6 = Sunday
    df_trips['date'] = df_trips['start_time'].dt.date
    daily_distance = df_trips.groupby(['vehicle_id', 'date', 'weekday'])['distance_km'].sum().reset_index()

    # Group by weekday and prepare boxplot data
    boxplot_data = daily_distance.groupby('weekday')['distance_km'].apply(list)

    # Define plot size
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    height = h_169 * 1.5  # Increase height for better readability
    scaling_factor = textwidth / 10

    fig, ax = plt.subplots(figsize=(textwidth, height))
    weekday_colors = [
        dp.colors['TUMBlue4'], dp.colors['TUMBlue3'], dp.colors['TUMBlue2'], 
        dp.colors['TUMBlue1'], dp.colors['TUMGray3'], dp.colors['TUMGray2'], 
        dp.colors['TUMGray1']
    ]

    # Create a boxplot with custom settings
    bp = ax.boxplot(
        boxplot_data,
        patch_artist=True,  # Color the boxes
        showmeans=True,  # Show mean value
        meanline=True,  # Use a line for the mean
        meanprops=dict(linestyle='--', linewidth=2 * scaling_factor, color=dp.colors['TUMOrange']),
        medianprops=dict(color=dp.colors['TUMGreen1']),
        showfliers=True,  # Show outliers
        flierprops=dict(marker='o', color='red', alpha=0.5),
        whiskerprops=dict(color=dp.colors['TUMBlack'], linewidth=1 * scaling_factor),
        capprops=dict(color=dp.colors['TUMBlack'], linewidth=1 * scaling_factor),
        widths=0.55 * scaling_factor,
        positions=range(len(boxplot_data)),
        labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )

    # Assign colors to the boxplots
    for box, color in zip(bp['boxes'], weekday_colors):
        box.set_facecolor(color)

    # Define legend handles
    median_line = mlines.Line2D([], [], color=dp.colors['TUMGreen1'], label='Median', linewidth=2 * scaling_factor)
    mean_line = mlines.Line2D([], [], color=dp.colors['TUMOrange'], label='Mean', linestyle='--', linewidth=2 * scaling_factor)
    plt.legend(handles=[median_line, mean_line], loc='upper right', fontsize=14 * scaling_factor, frameon=True)

    # Annotate mean and median values next to the boxes
    for i, (mean_feature, median_feature, distances) in enumerate(zip(bp['means'], bp['medians'], boxplot_data)):
        distances_series = pd.Series(distances)  # Convert list to Series
        avg_value = distances_series.mean()
        median_value = distances_series.median()
        x_mean, y_mean = mean_feature.get_xydata()[1]
        ax.text(x_mean - 0.75 * scaling_factor, y_mean, f'{avg_value:.2f}', horizontalalignment='center',
                color=dp.colors['TUMGreen1'], fontsize=10 * scaling_factor, bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=dp.colors['TUMOrange'], edgecolor=dp.colors['TUMOrange']))
        x_median, y_median = median_feature.get_xydata()[1]
        ax.text(x_median + 0.25 * scaling_factor, y_median, f'{median_value:.2f}', horizontalalignment='center',
                color=dp.colors['TUMBlack'], fontsize=10 * scaling_factor, bbox=dict(boxstyle='round', pad=0.3 * scaling_factor, facecolor=dp.colors['TUMWhite'], edgecolor=dp.colors['TUMBlack']))

    ax.set_xlabel('Weekday', fontsize=14 * scaling_factor)
    ax.set_ylabel('Distance per Truck / km', fontsize=14 * scaling_factor)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')


    # Print the current font settings
    print("Current font family:", plt.rcParams['font.family'])
    print("Current font size:", plt.rcParams['font.size'])
    print("Current font weight:", plt.rcParams['font.weight'])
    print("Current font style:", plt.rcParams['font.style'])

    for spine in plt.gca().spines.values():
        spine.set_edgecolor(dp.colors['TUMBlack'])
        spine.set_linewidth(1 * scaling_factor)

    ax.set_ylim(bottom=0)
    plt.xticks(rotation=90, fontsize=10 * scaling_factor)
    plt.yticks(fontsize=10 * scaling_factor)
    plt.tight_layout()
    plt.savefig('data/output/figures/temporal_pattern/fleet_truck_weekday_boxplots.svg', bbox_inches='tight')
    plt.show()


def plot_trip_duration_distance_histogram(df_trips, max_distance=400, max_duration=5):
    """
    Plot histograms for trip durations and distances.
    """
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    doc_font = {'fontname':'Arial'}
    print(plt.style)
    # Print the current font settings
    print("Current font family:", plt.rcParams['font.family'])
    print("Current font size:", plt.rcParams['font.size'])
    print("Current font weight:", plt.rcParams['font.weight'])
    print("Current font style:", plt.rcParams['font.style'])
    print('\n')

    gf_dis = GridFigure(1, 2, textwidth, h_169, wspace=0.25, constrained_layout=False)
    
    sns.histplot(data=df_trips.loc[df_trips.duration_h <= max_duration], 
                 x='duration_h',
                 bins=40,
                 element="step", 
                 fill=False,
                 common_norm=False, 
                 ax=gf_dis.axes_list[0])

    sns.histplot(data=df_trips.loc[df_trips.distance_km <= max_distance], 
                 x="distance_km",
                 bins=40,
                 element="step", 
                 fill=False,
                 common_norm=False, 
                 ax=gf_dis.axes_list[1])

    gf_dis.axes_list[0].set_title('Duration Histogram', **doc_font)
    gf_dis.axes_list[1].set_title('Distance Histogram')
    gf_dis.axes_list[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    gf_dis.axes_list[1].xaxis.set_major_locator(ticker.MultipleLocator(100))
    gf_dis.axes_list[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    gf_dis.axes_list[1].xaxis.set_minor_locator(ticker.MultipleLocator(25))
    gf_dis.axes_list[0].yaxis.set_minor_locator(ticker.MultipleLocator(25))
    gf_dis.axes_list[1].yaxis.set_minor_locator(ticker.MultipleLocator(100))

    # Remove the legend from the first axis if it exists
    legend_0 = gf_dis.axes_list[0].get_legend()
    if legend_0:
        legend_0.remove()

    # Get and set the legend for the second axis
    legend_1 = gf_dis.axes_list[1].get_legend()
    if legend_1:
        legend_1.set(title='fleet')

    gf_dis.axes_list[0].grid(which='major', alpha=alpha_major)
    gf_dis.axes_list[1].grid(which='major', alpha=alpha_major)
    gf_dis.axes_list[0].grid(which='minor', alpha=alpha_minor)
    gf_dis.axes_list[1].grid(which='minor', alpha=alpha_minor)

    gf_dis.axes_list[0].set(xlim=(0, max_duration), ylim=(0, 800), xlabel='Duration / h', ylabel='Count')
    gf_dis.axes_list[1].set(xlim=(0, max_distance), ylim=(0, 2500), xlabel='Distance / km', ylabel='')
    gf_dis.axes_list[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda t, pos: f"{int(t):,}"))


    # Print the current font settings
    print("Current font family:", plt.rcParams['font.family'])
    print("Current font size:", plt.rcParams['font.size'])
    print("Current font weight:", plt.rcParams['font.weight'])
    print("Current font style:", plt.rcParams['font.style'])

    setFontSize(gf_dis.axes_list, 9)
    plt.savefig('data/output/figures/temporal_pattern/trip_duration_and_distance.svg', bbox_inches='tight')
    plt.show()

# Farbe definieren
indicator_color = 'black'

def plot_weekly_distances(distances_km):
    """
    Plot the weekly distances covered by the fleet as a bar plot.
    """
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth

    fig, ax = plt.subplots(figsize=(textwidth, h_169))
    indicator_level = 70000

    # Alle Balken blau
    bar_colors = ['#1f77b4'] * len(distances_km)

    distances_km.plot(kind='bar', stacked=True, color=bar_colors, xlabel='', ylabel='Distance / km', ax=ax)
    
    # Indicator line
    ax.axhline(y=indicator_level, color=indicator_color, linestyle='--', linewidth=1.5)
    ax.axvline(x=0, ymin=0, ymax=(indicator_level / 80000), color=indicator_color, linestyle='--', linewidth=1.5)
    ax.axvline(x=len(distances_km) - 1, ymin=0, ymax=(indicator_level / 80000), color=indicator_color, linestyle='--', linewidth=1.5)
    
    # Calculate and set x-ticks for years
    index_as_timestamp = distances_km.index.to_timestamp()
    years = pd.to_datetime(index_as_timestamp).year.unique()
    year_ticks = []
    for year in years:
        year_weeks = index_as_timestamp.to_period('W').strftime('%Y').astype(int) == year
        mid_week = len(distances_km[year_weeks]) // 2
        year_ticks.append(np.where(year_weeks)[0][mid_week])

    x_week_ticks = np.arange(len(distances_km))
    ax.set(ylim=(0, 80000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5000))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t, pos: f"{int(t):,}"))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(x_week_ticks))
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda t, pos: f"{index_as_timestamp[t].day:02d}-{index_as_timestamp[t].month:02d}"))
    ax.xaxis.set_major_locator(ticker.FixedLocator(year_ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(years[pos]) if pos < len(years) else ''))

    # Adjust rotation of x-ticks and add gridlines
    ax.tick_params(axis="x", which="major", pad=20, labelrotation=0)
    ax.tick_params(axis="x", which="minor", pad=4, labelrotation=90)
    ax.grid(axis='y', which='major', linestyle='-', linewidth=0.75, color='gray', alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

    # Adding dashed line to separate years
    if len(year_ticks) > 1:
        for tick in year_ticks[1:]:
            ax.axvline(x=tick - 3.5, color='green', linestyle='--')

    ax.set_xlabel('Weeks', fontsize=10)
    ax.set_ylabel('Distance / km', fontsize=10)
    plt.tight_layout()
    plt.savefig('data/output/figures/temporal_pattern/weekly_distance_barplot.svg', bbox_inches='tight')
    plt.show()


def plot_drive_vs_pause_by_weekday(df_trips):
    """
    Plot the average drive and pause durations by weekday.
    """
    avg_times = dp.calculate_drive_pause_by_weekday(df_trips)

    # Prepare data for plotting
    plot_data = pd.melt(avg_times, id_vars=['weekday'], value_vars=['drive_time', 'pause_time', 'remaining_pause_at_start', 'remaining_pause_at_end'],
                        var_name='type', value_name='hours')

    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth

    # Create the plot
    fig, ax = plt.subplots(figsize=(textwidth, h_169))
    sns.barplot(x='weekday', y='hours', hue='type', data=plot_data, palette=[dp.colors['TUMBlue1'], dp.colors['TUMOrange'], dp.colors['TUMGreen3'], dp.colors['TUMGray1']], ax=ax)

    # Customize the plot
    ax.set_xlabel('Weekday')
    ax.set_ylabel('Average Time (hours)')
    ax.set_title('Average Drive vs Pause Times by Weekday')
    ax.legend(title='Time Type', labels=['Drive Time', 'Pause Time', 'Remaining Pause at Start', 'Remaining Pause at End'])
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig('data/output/figures/temporal_pattern/drive_vs_pause_times_by_weekday.svg', bbox_inches='tight')
    plt.show()


# # Options - Setze Optionen für Detailansichten und Indikatorlevel für Geschwindigkeitsvisualisierungen
# idxmin_detail, idxmax_detail = 2400, 2700
# indicator_level_kmh, max_speed = 75, 80

# # Prepare figure
# gf = GridFigure(1, 2, textwidth, h_169/2, wspace=0.1)

# # Weise die Achsenobjekte aus der GridFigure lokalen Variablen zu
# speed_overview_ax = gf.axes_list[0]
# speed_detail_ax = gf.axes_list[1]

# # Plot
# df_speed.plot(y='kmh', ax=speed_overview_ax)
# df_speed.plot(y='kmh', ax=speed_detail_ax)

# # Erstelle Zwillingsachsen für das Plotten von HDOP-Daten auf denselben x-Achsen
# # HDOP steht für "Horizontal Dilution of Precision" und ist ein Maß für die Genauigkeit, mit der ein GPS-Empfänger (Global Positioning System) den Standort horizontal (in der Ebene, in der Regel in Breiten- und Längengrad) bestimmen kann.
# # HDOP ist einer von mehreren "Dilution of Precision" (DOP) Indikatoren, die die Qualität der Satellitenkonstellationen und deren Einfluss auf die Positionsbestimmung bewerten.
# # Niedriger HDOP-Wert: Dies deutet auf eine bessere Genauigkeit der Positionsangaben hin. Ein niedriger HDOP bedeutet, dass die GPS-Satelliten, die zur Berechnung der Position herangezogen werden, günstig verteilt sind.
# # Hoher HDOP-Wert: Ein hoher Wert zeigt an, dass die Genauigkeit der Standortbestimmung durch eine ungünstige Verteilung der Satelliten beeinträchtigt wird
# speed_overview_ax_hdop = speed_overview_ax.twinx()
# speed_detail_ax_hdop = speed_detail_ax.twinx()

# df_speed.plot(y='hdop', ax=speed_overview_ax_hdop, color=colors_locations['driving'], linestyle='--')
# df_speed.plot(y='hdop', ax=speed_detail_ax_hdop, color=colors_locations['driving'], linestyle='--')

# for ax in gf.axes_list:
#     ax.plot([df_speed.index[idxmin_detail],df_speed.index[idxmax_detail]], [indicator_level_kmh, indicator_level_kmh],
#             "--v", linewidth=1, markersize=5) #  Zoom indicator

# # Format figure
# speed_overview_ax.set(xlim=(0,df_speed.index.max()))
# speed_detail_ax.set(xlim=(df_speed.index[idxmin_detail],df_speed.index[idxmax_detail]))

# speed_overview_ax.set(
#     xlabel='time / s',
#     ylabel='speed / km/h',
#     title='full trip',
#     ylim=(0,max_speed),
#     xlim=(0,800),
# )

# speed_detail_ax.set(
#     xlabel='time / s',
#     title='acceleration ramp',
#     ylim=(0,max_speed),
#     yticklabels=[],
#     xlim=(240,280),
# )

# # Konfiguriere die Achsen der HDOP-Plots, entferne y-Achsenbeschriftungen und setze Grenzen
# speed_overview_ax_hdop.set(
#     ylabel='',
#     yticklabels=[],
#     ylim=(0,1.5)
# )

# speed_detail_ax_hdop.set(
#     ylabel='hdop / m',
#     ylim=(0,1.5),
# )

# # Setze Tick-Intervalle für die Haupt- und Nebenachsen für eine verbesserte Lesbarkeit
# speed_overview_ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
# speed_overview_ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))

# speed_detail_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# speed_detail_ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

# speed_overview_ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
# speed_overview_ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

# speed_detail_ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
# speed_detail_ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

# # Format figure
# speed_overview_ax.grid(which='major', alpha=alpha_major)
# speed_overview_ax.grid(which='minor', alpha=alpha_minor)
# speed_detail_ax.grid(which='major', alpha=alpha_major)
# speed_detail_ax.grid(which='minor', alpha=alpha_minor)

# handles_1, labels_1 = speed_detail_ax_hdop.get_legend_handles_labels()
# handles_2, labels_2 = speed_detail_ax.get_legend_handles_labels()

# speed_overview_ax_hdop.get_legend().remove()
# speed_detail_ax_hdop.get_legend().remove()

# speed_overview_ax.get_legend().remove()
# speed_detail_ax.get_legend().remove()

# speed_detail_ax.legend(handles = handles_1+handles_2, labels=['hdop', 'speed'], ncol=1, loc=(0.025,.5))

# # Rotiere die x-Achsen-Beschriftungen für bessere Lesbarkeit
# for ax_i in gf.axes_list:
#     for tick in ax_i.get_xticklabels():
#         tick.set(rotation=90)#,ha='right')

# # Set font sizes
# setFontSize(gf.axes_list, 9)

# plt.savefig('figures/paper/speed_profile.svg', bbox_inches='tight')

# ------------------------------------------------------------------------------
#                              SPATIAL PATTERN
# ------------------------------------------------------------------------------


def plot_rest_time_kde(df_stops):
    """
    Plot KDEs for rest times segmented by location.
    """
    colors_locations_2 = dp.colors_locations_2

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
        sns.kdeplot(data=data, x='rest_time_h', hue='location', bw_adjust=bw_adjust, palette=colors_locations_2, cut=0, common_norm=False, ax=ax)

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

    setFontSize(gf_loc_axes, 9)
    handles = [genLineLegendHandle(c) for c in colors_locations_2.values()]
    gf_loc.legend_ax.legend(handles=handles[:-1], labels=list(colors_locations_2.keys())[:-1], ncol=3, loc="center")

    gf_loc.fig.savefig(f'data/output/figures/spatial_pattern/rest_time_global.svg', bbox_inches='tight')
    gf_loc_2.fig.savefig(f'data/output/figures/spatial_pattern/rest_time_detail.svg', bbox_inches='tight')


# ------------------------------------------------------------------------------
#                              DATA QUALITY
# ------------------------------------------------------------------------------


def plot_data_quality_violin(df_trips):
    """
    Plot violin plots to show data quality metrics.
    """
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth

    # Ensure 'distance_km' and 'track_gap_km' columns are present
    if 'distance_km' not in df_trips.columns:
        df_trips['distance_km'] = df_trips['distance'] / 1000
    if 'track_gap_km' not in df_trips.columns:
        df_trips['track_gap_km'] = df_trips['track_gap'] / 1000
    if 'ratio' not in df_trips.columns:
        df_trips['ratio'] = df_trips['track_gap'] / df_trips['distance']

    gf = GridFigure(1, 3, textwidth * .75, h_169 * .75, wspace=0.8, constrained_layout=False)

    df_track_gap = df_trips.groupby('vehicle_id').sum() # TODO this line breaks (TypeError: datetime64 type does not support sum operations)

    # Remove NaN and inf values
    df_track_gap = df_track_gap.replace([np.inf, -np.inf], np.nan).dropna(subset=['distance_km', 'track_gap_km', 'ratio'])

    distances_ax = gf.axes_list[0]
    gaps_ax = gf.axes_list[1]
    ratios_ax = gf.axes_list[2]

    # Create violin plots
    sns.violinplot(data=df_track_gap.distance_km.values, ax=distances_ax, inner='stick', cut=0, color='#A0A0A0')
    sns.violinplot(data=df_track_gap.track_gap_km.values, ax=gaps_ax, inner='stick', color='#A0A0A0')
    sns.violinplot(data=df_track_gap.ratio.values, ax=ratios_ax, inner='stick', color='#A0A0A0')

    distances_ax.set(
        ylabel="recorded distance / km",
        ylim=(0, 5*10**4)
    )

    gaps_ax.set(
        ylabel="track gap / km",
        ylim=(0, 2.5*10**3)
    )

    ratios_ax.set(
        ylabel="gap-to-distance ratio",
        ylim=(0, 0.7),
        yticks=np.arange(0,0.9, 0.2)
    )

    for ax in gf.axes_list:
        ax.grid("y", which='major', alpha=alpha_major)
        ax.xaxis.labelpad = 10
        ax.set(xticks=[])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t,pos: f"{t:,.1f}".replace('.0','')))

    setFontSize(gf.axes_list, 9)
    plt.savefig('data/output/figures/data_quality.svg', bbox_inches='tight')
    plt.show()


def plot_fleet_occupation(df_occ, truck_day):

    # Prepare figure
    textwidth = 159.2 / 25.4  # Convert text width from mm to inches
    h_169 = 9 / 16 * textwidth
    gridfig = GridFigure(1,2,textwidth, h_169, wspace=0.35, hspace=.5, legend_mode="above", constrained_layout=False, ratios={'height':[1], 'width':[1, 2]})
    ax = gridfig.axes_list[0]
    LOCATIONS = dp.LOCATIONS
    
    color_list = list(colors.values())
    colors_fleets = {i: color_list[i - 1] for i in range(1, 5)}
    colors_locations = {loc: color_list[i + 4] for i, loc in enumerate([*LOCATIONS, 'other_area', 'driving'])}
    colors_locations_2 = {loc.replace('_', ' '): color_list[i + 4] for i, loc in enumerate([*LOCATIONS, 'other_area', 'driving'])}

    # Plot
    # df_occ: absolute numbers per vehicle in each occupation 
    plot_data = (df_occ.transpose() / df_occ.transpose().sum()).transpose() #relative values per vehicle
    plot_data = plot_data[['driving','home_base','industrial_area','other_area','rest_area','service_area_fuel']] #rearange columns
    plot_data = plot_data.fillna(0)
    plot_data = plot_data.mean().to_frame().T
    plot_data.plot.bar(ax = ax, stacked=True, color=colors_locations, width=0.5)

    # Adjust figure
    ax.set(
        ylim=(0,1),
        yticks=np.arange(0,1.04,0.1),
        xlabel="fleet",
        ylabel="share of vehicle status by fleet"
    )

    ax.grid(axis="y", which='major', alpha=alpha_major)
    ax.grid(axis="y", which='minor', alpha=alpha_minor)

    # Create legend
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t, pos: f"{round(t*100)}"+"%"))# + u'\u2006%'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.05))

    handles,labels = ax.get_legend_handles_labels()
    gridfig.legend_ax.axis("off")
    gridfig.legend_ax.legend(handles = handles, labels=[l.replace("_"," ") for l in labels], ncol=3, loc="center")
    ax.get_legend().remove()

    # Adjust fonts
    ax = gridfig.axes_list[1]

    # Plot
    plot_data = (truck_day / truck_day.sum()).transpose()
    plot_data.plot.bar(ax=ax, stacked=True, color=colors_locations)

    # Adjust figure
    ax.set(
        ylim=(0,1),
        yticks=np.arange(0,1.04,0.1),
        xlabel="hour",
        ylabel="share of vehicle status by time"
    )

    # Create legend
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t, pos: f"{round(t*100)}%"))
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
    plt.savefig('data/output/own_figures/temporal_pattern/fleet_occupation.svg')
    plt.show()


def daily_arrival_pattern(dist_arrival_daily, dist_arrival):
    """
    Plot the daily arrival pattern of trucks and the average number of trucks arriving in the last hour below.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))

    # Plot the values of each day in the first subplot
    for idx, (date, row) in enumerate(dist_arrival_daily.iterrows()):
        axes[0].plot(row.index, row.values, label=f'Day {idx + 1}', alpha=0.5)  # Adjust alpha for transparency

    # Set the x-axis to represent the 24 hours for the first subplot
    axes[0].set_xticks(dist_arrival_daily.columns)

    # Add labels and title for the first subplot
    axes[0].set_xlabel('Hour of the Day')
    axes[0].set_ylabel('Distance Driven')
    axes[0].set_title('Distance Driven by Trucks Arriving in the Last Hour (Daily)')

    # Optionally, add a legend for the first subplot (commented out to avoid cluttering the plot)
    # axes[0].legend(loc='upper right', fontsize='small')

    # Plot the average number of trucks arriving in the last hour in the second subplot
    axes[1].plot(dist_arrival.columns, dist_arrival.loc['avg_num_trucks_per_day'], marker='o', linestyle='-', color='b')

    # Set the x-axis to represent the 24 hours for the second subplot
    axes[1].set_xticks(dist_arrival.columns)

    # Add labels and title for the second subplot
    axes[1].set_xlabel('Hour of the Day')
    axes[1].set_ylabel('Average Number of Trucks')
    axes[1].set_title('Average Number of Trucks Arriving in the Last Hour (Daily)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.savefig('data/output/own_figures/temporal_pattern/daily_arrival_distances.svg')
    plt.show()

