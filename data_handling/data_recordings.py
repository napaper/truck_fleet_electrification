"""
Truck Data Recording Analysis Script

This script analyzes truck recording periods and activity patterns from GPS tracking data.
It provides functions to analyze how many trucks are recording data at any given time,
both overall and per freight forwarder.

IMPORTANT NOTE: This script is NOT required for the main truck fleet electrification analyses.
It is provided for additional data exploration and validation purposes only.
The main analysis workflow can be executed without this script.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


def analyze_truck_recording_periods(csv_path='input/stations/tracks.csv'):
    """
    Analyze truck recording periods to determine active trucks over time.
    
    This function processes GPS tracking data to identify when each truck was actively
    recording data and calculates daily counts of active trucks, both overall and
    broken down by freight forwarder.
    
    Parameters
    ----------
    csv_path : str, default='input/stations/tracks.csv'
        Path to the CSV file containing truck tracking data
        
    Returns
    -------
    tuple
        A tuple containing:
        - active_trucks_df : pandas.DataFrame
            DataFrame with one row per truck containing first/last recording times
        - recording_counts_df : pandas.DataFrame  
            Daily counts of total active trucks
        - recording_counts_by_ff_df : pandas.DataFrame
            Daily counts of active trucks per freight forwarder
        - first_last_dates : tuple
            Tuple with (first_date, last_date) in the dataset
            
    Raises
    ------
    FileNotFoundError
        If the specified CSV file cannot be found
    ValueError
        If required columns are missing from the data
    """
    # Load and validate input data
    try:
        df_tours = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Validate required columns exist
    required_columns = ['start_time', 'stop_time', 'vehicle_id']
    missing_columns = [col for col in required_columns if col not in df_tours.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert timestamps to datetime with UTC timezone
    df_tours['start_time'] = pd.to_datetime(df_tours['start_time'], format='mixed', utc=True)
    df_tours['stop_time'] = pd.to_datetime(df_tours['stop_time'], format='mixed', utc=True)
    
    # Ensure freight_forwarder information is available
    df_tours = _add_freight_forwarder_info(df_tours)
    
    # Calculate recording periods for each truck
    active_trucks = _calculate_truck_recording_periods(df_tours)
    
    # Generate daily activity counts
    recording_counts, recording_counts_by_ff = _generate_daily_counts(active_trucks)
    
    # Extract overall date range
    first_date = active_trucks['start_time'].min()
    last_date = active_trucks['stop_time'].max()
    first_last_dates = (first_date, last_date)
    
    return active_trucks, recording_counts, recording_counts_by_ff, first_last_dates


def _add_freight_forwarder_info(df_tours):
    """
    Add freight forwarder information to the tours dataframe.
    
    If freight_forwarder column is missing, attempts to load it from alternative
    data sources or creates a default value.
    
    Parameters
    ----------
    df_tours : pandas.DataFrame
        Tours dataframe that may be missing freight_forwarder column
        
    Returns
    -------
    pandas.DataFrame
        Tours dataframe with freight_forwarder column added
    """
    if 'freight_forwarder' in df_tours.columns:
        return df_tours
    
    # Attempt to load freight_forwarder from alternative source
    try:
        df_trips = pd.read_csv('input/stations/tracks_filtered.csv')
        vehicle_to_ff = df_trips[['vehicle_id', 'freight_forwarder']].drop_duplicates()
        df_tours = df_tours.merge(vehicle_to_ff, on='vehicle_id', how='left')
        print("Successfully loaded freight_forwarder data from tracks_filtered.csv")
    except Exception as e:
        print(f"Warning: Could not load freight_forwarder data: {e}")
        print("Creating default freight_forwarder column with value 1")
        df_tours['freight_forwarder'] = 1
    
    return df_tours


def _calculate_truck_recording_periods(df_tours):
    """
    Calculate the recording period for each truck.
    
    Parameters
    ----------
    df_tours : pandas.DataFrame
        Tours dataframe with start_time, stop_time, and vehicle_id columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per truck and recording period information
    """
    # Group by vehicle_id to find first start_time and last stop_time for each truck
    first_start = df_tours.groupby('vehicle_id')['start_time'].min().reset_index()
    last_stop = df_tours.groupby('vehicle_id')['stop_time'].max().reset_index()
    
    # Merge the two dataframes
    active_trucks = first_start.merge(last_stop, on='vehicle_id', suffixes=('_first', '_last'))
    
    # Add freight_forwarder information
    vehicle_freight_forwarder = df_tours.groupby('vehicle_id')['freight_forwarder'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    ).reset_index()
    
    active_trucks = active_trucks.merge(vehicle_freight_forwarder, on='vehicle_id', how='left')
    
    # Rename columns for clarity
    active_trucks.rename(columns={
        'start_time_first': 'start_time', 
        'stop_time_last': 'stop_time'
    }, inplace=True)
    
    return active_trucks


def _generate_daily_counts(active_trucks):
    """
    Generate daily counts of active trucks.
    
    Parameters
    ----------
    active_trucks : pandas.DataFrame
        DataFrame with truck recording periods
        
    Returns
    -------
    tuple
        (recording_counts, recording_counts_by_ff) - Daily counts overall and by freight forwarder
    """
    # Find the overall first and last dates in the dataset
    first_date = active_trucks['start_time'].min()
    last_date = active_trucks['stop_time'].max()
    
    # Create a date range from first to last date
    date_range = pd.date_range(start=first_date.date(), end=last_date.date(), freq='D')
    
    # Get unique freight forwarders
    freight_forwarders = sorted(active_trucks['freight_forwarder'].unique())
    
    # Count active trucks for each day
    active_counts = []
    active_counts_by_ff = []
    
    for date in date_range:
        # Convert date to datetime for comparison
        date_start = pd.Timestamp(date, tz='UTC')
        date_end = pd.Timestamp(date, tz='UTC') + pd.Timedelta(days=1)
        
        # Filter trucks active on this date
        active_on_date = active_trucks[
            (active_trucks['start_time'] < date_end) & 
            (active_trucks['stop_time'] >= date_start)
        ]
        
        # Count total active trucks
        total_active = len(active_on_date)
        active_counts.append({'date': date, 'active_trucks': total_active})
        
        # Count active trucks by freight forwarder
        for ff in freight_forwarders:
            ff_active = len(active_on_date[active_on_date['freight_forwarder'] == ff])
            active_counts_by_ff.append({
                'date': date,
                'freight_forwarder': ff,
                'active_trucks': ff_active
            })
    
    recording_counts = pd.DataFrame(active_counts)
    recording_counts_by_ff = pd.DataFrame(active_counts_by_ff)
    
    return recording_counts, recording_counts_by_ff


def plot_active_trucks(recording_counts, recording_counts_by_ff=None, 
                       first_last_dates=None, save_path=None):
    """
    Create visualization of truck recording activity over time.
    
    Generates a two-panel plot showing total active trucks and breakdown by
    freight forwarder over the recording period.
    
    Parameters
    ----------
    recording_counts : pandas.DataFrame
        DataFrame with 'date' and 'active_trucks' columns
    recording_counts_by_ff : pandas.DataFrame, optional
        DataFrame with 'date', 'freight_forwarder', and 'active_trucks' columns
    first_last_dates : tuple, optional
        Tuple with (first_date, last_date) for annotating the plot
    save_path : str, optional
        Path to save the plot image
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Total active trucks
    _plot_total_active_trucks(ax1, recording_counts, first_last_dates)
    
    # Plot 2: Active trucks by freight forwarder
    if recording_counts_by_ff is not None:
        _plot_trucks_by_freight_forwarder(ax2, recording_counts_by_ff)
    
    # Format x-axis date ticks and layout
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig


def _plot_total_active_trucks(ax, recording_counts, first_last_dates):
    """Helper function to plot total active trucks."""
    # Plot total active trucks
    ax.plot(recording_counts['date'], recording_counts['active_trucks'], 
            marker='o', linestyle='-', color='#0065BD', linewidth=2)
    
    # Add grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Total Number of Trucks Recording Data Over Time', fontsize=16)
    ax.set_ylabel('Number of Active Trucks', fontsize=14)
    ax.set_ylim(bottom=0)
    
    # Add annotations
    total_trucks = recording_counts['active_trucks'].max()
    ax.annotate(f'Maximum active trucks: {total_trucks}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    if first_last_dates:
        first_date, last_date = first_last_dates
        duration_days = (last_date - first_date).days
        ax.annotate(f'Recording period: {duration_days} days\n({first_date.date()} to {last_date.date()})', 
                    xy=(0.02, 0.85), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))


def _plot_trucks_by_freight_forwarder(ax, recording_counts_by_ff):
    """Helper function to plot trucks by freight forwarder."""
    # Default colors for freight forwarders
    colors = plt.cm.tab10.colors
    
    # Get list of freight forwarders
    freight_forwarders = sorted(recording_counts_by_ff['freight_forwarder'].unique())
    
    # Plot each freight forwarder
    for i, ff in enumerate(freight_forwarders):
        ff_data = recording_counts_by_ff[recording_counts_by_ff['freight_forwarder'] == ff]
        ff_label = f'FF {ff}'
        
        ax.plot(ff_data['date'], ff_data['active_trucks'], 
                marker='o', linestyle='-', color=colors[i % len(colors)], 
                linewidth=2, label=ff_label)
    
    # Add grid, legend, and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('Active Trucks by Freight Forwarder', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Number of Active Trucks', fontsize=14)
    ax.set_ylim(bottom=0)


def analyze_truck_activity(csv_path='input/stations/tours.csv', save_plot=True):
    """
    Main function to analyze and visualize truck recording activity.
    
    This function orchestrates the complete analysis workflow, including data loading,
    processing, visualization, and reporting of truck recording patterns.
    
    Parameters
    ----------
    csv_path : str, default='input/stations/tours.csv'
        Path to the CSV file containing truck tour data
    save_plot : bool, default=True
        Whether to save the generated plot to a file
        
    Returns
    -------
    tuple
        A tuple containing:
        - active_trucks_df : pandas.DataFrame
            DataFrame with truck recording periods
        - recording_counts_df : pandas.DataFrame
            Daily counts of active trucks
        - recording_counts_by_ff_df : pandas.DataFrame
            Daily counts by freight forwarder
        - figure : matplotlib.figure.Figure
            The generated visualization
    """
    # Execute the complete analysis workflow
    active_trucks, recording_counts, recording_counts_by_ff, first_last_dates = analyze_truck_recording_periods(csv_path)
    
    # Print summary statistics
    _print_analysis_summary(active_trucks, recording_counts, first_last_dates)
    
    # Generate and save visualization
    save_path = 'output/figures/active_trucks_over_time.svg' if save_plot else None
    fig = plot_active_trucks(recording_counts, recording_counts_by_ff, 
                            first_last_dates, save_path)
    
    # Calculate and display recording duration statistics
    _analyze_recording_durations(active_trucks)
    
    return active_trucks, recording_counts, recording_counts_by_ff, fig


def _print_analysis_summary(active_trucks, recording_counts, first_last_dates):
    """Print summary statistics from the analysis."""
    print(f"Total number of trucks in dataset: {len(active_trucks)}")
    print(f"First recording date: {first_last_dates[0].date()}")
    print(f"Last recording date: {first_last_dates[1].date()}")
    print(f"Total recording period: {(first_last_dates[1] - first_last_dates[0]).days} days")
    
    # Print active trucks at different time points
    print("\nActive trucks over time:")
    for percentile in [0, 25, 50, 75, 100]:
        date_idx = int(len(recording_counts) * percentile / 100)
        if date_idx < len(recording_counts):
            date_entry = recording_counts.iloc[date_idx]
            print(f"  {date_entry['date'].date()}: {date_entry['active_trucks']} trucks")
    
    # Print trucks by freight forwarder
    print("\nTrucks by freight forwarder:")
    for ff in sorted(active_trucks['freight_forwarder'].unique()):
        ff_count = len(active_trucks[active_trucks['freight_forwarder'] == ff])
        print(f"  FF {ff}: {ff_count} trucks")


def _analyze_recording_durations(active_trucks):
    """Analyze and display recording duration statistics."""
    # Calculate recording duration for each truck
    active_trucks['recording_duration_days'] = (
        active_trucks['stop_time'] - active_trucks['start_time']).dt.total_seconds() / (24*60*60)
    
    print("\nTruck recording duration statistics:")
    print(f"  Mean duration: {active_trucks['recording_duration_days'].mean():.2f} days")
    print(f"  Median duration: {active_trucks['recording_duration_days'].median():.2f} days")
    print(f"  Min duration: {active_trucks['recording_duration_days'].min():.2f} days")
    print(f"  Max duration: {active_trucks['recording_duration_days'].max():.2f} days")
    
    # Analysis by freight forwarder
    print("\nRecording duration by freight forwarder:")
    for ff in sorted(active_trucks['freight_forwarder'].unique()):
        ff_trucks = active_trucks[active_trucks['freight_forwarder'] == ff]
        print(f"  FF {ff}:")
        print(f"    Mean duration: {ff_trucks['recording_duration_days'].mean():.2f} days")
        print(f"    Median duration: {ff_trucks['recording_duration_days'].median():.2f} days")


def count_active_trucks_at_time(active_trucks, target_time):
    """
    Count how many trucks were recording at a specific time.
    
    Parameters
    ----------
    active_trucks : pandas.DataFrame
        DataFrame with truck recording periods
    target_time : str or datetime
        The specific time to check (can be string or datetime object)
        
    Returns
    -------
    int
        Number of trucks actively recording at the specified time
    """
    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time, utc=True)
        
    active_count = len(active_trucks[
        (active_trucks['start_time'] <= target_time) & 
        (active_trucks['stop_time'] >= target_time)
    ])
    
    return active_count


def count_active_trucks_by_ff_at_time(active_trucks, target_time):
    """
    Count active trucks by freight forwarder at a specific time.
    
    Parameters
    ----------
    active_trucks : pandas.DataFrame
        DataFrame with truck recording periods
    target_time : str or datetime
        The specific time to check
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with freight_forwarder and count columns
    """
    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time, utc=True)
        
    # Filter to active trucks at target time
    active_at_time = active_trucks[
        (active_trucks['start_time'] <= target_time) & 
        (active_trucks['stop_time'] >= target_time)
    ]
    
    # Group and count by freight forwarder
    return active_at_time.groupby('freight_forwarder').size().reset_index(name='count')


# Main execution block
if __name__ == "__main__":
    # Execute the main analysis
    active_trucks, recording_counts, recording_counts_by_ff, fig = analyze_truck_activity()
    
    # Example usage for point-in-time analysis
    target_time = "2021-09-15 12:00:00"
    
    # Count total active trucks at specific time
    active_count = count_active_trucks_at_time(active_trucks, target_time)
    print(f"\nTrucks recording at {target_time}: {active_count}")
    
    # Count by freight forwarder at specific time
    active_by_ff = count_active_trucks_by_ff_at_time(active_trucks, target_time)
    print(f"\nTrucks recording at {target_time} by freight forwarder:")
    
    for _, row in active_by_ff.iterrows():
        ff = row['freight_forwarder']
        print(f"  FF {ff}: {row['count']} trucks")