import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def analyze_truck_recording_periods(csv_path='input/stations/tracks.csv'):
    """
    Analyze how many trucks are recording data at any given time,
    both overall and per freight forwarder.
    
    Parameters:
    -----------
    csv_path : str
        Path to the tours.csv file
    
    Returns:
    --------
    tuple
        (active_trucks_df, recording_counts_df, recording_counts_by_ff_df, first_last_dates)
        - active_trucks_df: DataFrame with a row per truck and columns for first/last recording times
        - recording_counts_df: DataFrame with counts of total active trucks per day
        - recording_counts_by_ff_df: DataFrame with counts of active trucks per day by freight forwarder
        - first_last_dates: tuple with (first_date, last_date) in the dataset
    """
    # Read the tours data
    df_tours = pd.read_csv(csv_path)
    
    # Convert timestamps to datetime
    df_tours['start_time'] = pd.to_datetime(df_tours['start_time'], format='mixed', utc=True)
    df_tours['stop_time'] = pd.to_datetime(df_tours['stop_time'], format='mixed', utc=True)
    
    # Make sure we have freight_forwarder information
    if 'freight_forwarder' not in df_tours.columns:
        # If freight_forwarder isn't in tracks.csv, try to get it from another source
        try:
            # Try loading from trips data which might have freight_forwarder
            df_trips = pd.read_csv('input/stations/tracks_filtered.csv')
            
            # Get a mapping of vehicle_id to freight_forwarder
            vehicle_to_ff = df_trips[['vehicle_id', 'freight_forwarder']].drop_duplicates()
            
            # Merge with tours to add freight_forwarder
            df_tours = df_tours.merge(vehicle_to_ff, on='vehicle_id', how='left')
        except Exception as e:
            print(f"Warning: Could not load freight_forwarder data: {e}")
            # Create a default freight_forwarder column
            df_tours['freight_forwarder'] = 1
    
    # Group by vehicle_id to find first start_time and last stop_time for each truck
    first_start = df_tours.groupby('vehicle_id')['start_time'].min().reset_index()
    last_stop = df_tours.groupby('vehicle_id')['stop_time'].max().reset_index()
    
    # Merge the two dataframes
    active_trucks = first_start.merge(last_stop, on='vehicle_id', suffixes=('_first', '_last'))
    
    # Add freight_forwarder information to active_trucks
    # Get the most common freight_forwarder for each vehicle
    vehicle_freight_forwarder = df_tours.groupby('vehicle_id')['freight_forwarder'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    ).reset_index()
    
    # Merge with active_trucks
    active_trucks = active_trucks.merge(vehicle_freight_forwarder, on='vehicle_id', how='left')
    
    # Rename columns for clarity after merge
    active_trucks.rename(columns={'start_time_first': 'start_time', 
                                 'stop_time_last': 'stop_time'}, inplace=True)
    
    # Find the overall first and last dates in the dataset
    first_date = active_trucks['start_time'].min()
    last_date = active_trucks['stop_time'].max()
    
    # Create a date range from first to last date
    date_range = pd.date_range(start=first_date.date(), end=last_date.date(), freq='D')
    
    # Get unique freight forwarders
    freight_forwarders = sorted(active_trucks['freight_forwarder'].unique())
    
    # Count active trucks for each day (total and per freight forwarder)
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
    
    return active_trucks, recording_counts, recording_counts_by_ff, (first_date, last_date)

def plot_active_trucks(recording_counts, recording_counts_by_ff=None, 
                       first_last_dates=None, save_path=None):
    """
    Plot the number of trucks actively recording data over time,
    both overall and per freight forwarder.
    
    Parameters:
    -----------
    recording_counts : pandas.DataFrame
        DataFrame with 'date' and 'active_trucks' columns
    recording_counts_by_ff : pandas.DataFrame, optional
        DataFrame with 'date', 'freight_forwarder', and 'active_trucks' columns
    first_last_dates : tuple, optional
        Tuple with (first_date, last_date) for annotating the plot
    save_path : str, optional
        Path to save the plot image
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Total active trucks
    ax1.plot(recording_counts['date'], recording_counts['active_trucks'], 
            marker='o', linestyle='-', color='#0065BD', linewidth=2)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set title and labels
    ax1.set_title('Total Number of Trucks Recording Data Over Time', fontsize=16)
    ax1.set_ylabel('Number of Active Trucks', fontsize=14)
    
    # Set y-axis to start from 0
    ax1.set_ylim(bottom=0)
    
    # Add annotation for total number of trucks
    total_trucks = recording_counts['active_trucks'].max()
    ax1.annotate(f'Maximum active trucks: {total_trucks}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Plot 2: Active trucks by freight forwarder
    if recording_counts_by_ff is not None:
        # Default colors for freight forwarders
        colors = plt.cm.tab10.colors
        
        # Get list of freight forwarders
        freight_forwarders = sorted(recording_counts_by_ff['freight_forwarder'].unique())
        
        # Plot each freight forwarder
        for i, ff in enumerate(freight_forwarders):
            ff_data = recording_counts_by_ff[recording_counts_by_ff['freight_forwarder'] == ff]
            
            # Use FF number directly as label
            ff_label = f'FF {ff}'
                
            ax2.plot(ff_data['date'], ff_data['active_trucks'], 
                    marker='o', linestyle='-', color=colors[i % len(colors)], 
                    linewidth=2, label=ff_label)
        
        # Add grid and legend
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', fontsize=12)
        
        # Set title and labels
        ax2.set_title('Active Trucks by Freight Forwarder', fontsize=16)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('Number of Active Trucks', fontsize=14)
        
        # Set y-axis to start from 0
        ax2.set_ylim(bottom=0)
    
    if first_last_dates:
        first_date, last_date = first_last_dates
        duration_days = (last_date - first_date).days
        ax1.annotate(f'Recording period: {duration_days} days\n({first_date.date()} to {last_date.date()})', 
                    xy=(0.02, 0.85), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Format x-axis date ticks
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    
    return fig

def analyze_truck_activity(csv_path='input/stations/tours.csv', save_plot=True):
    """
    Main function to analyze and visualize truck recording activity.
    
    Parameters:
    -----------
    csv_path : str
        Path to the tours.csv file
    save_plot : bool
        Whether to save the plot to a file
    
    Returns:
    --------
    tuple
        (active_trucks_df, recording_counts_df, recording_counts_by_ff_df, figure)
    """
    # Run the analysis
    active_trucks, recording_counts, recording_counts_by_ff, first_last_dates = analyze_truck_recording_periods(csv_path)
    
    print(f"Total number of trucks in dataset: {len(active_trucks)}")
    print(f"First recording date: {first_last_dates[0].date()}")
    print(f"Last recording date: {first_last_dates[1].date()}")
    print(f"Total recording period: {(first_last_dates[1] - first_last_dates[0]).days} days")
    
    # Print the number of active trucks at different points in time
    print("\nActive trucks over time:")
    for percentile in [0, 25, 50, 75, 100]:
        date_idx = int(len(recording_counts) * percentile / 100)
        if date_idx < len(recording_counts):
            date_entry = recording_counts.iloc[date_idx]
            print(f"  {date_entry['date'].date()}: {date_entry['active_trucks']} trucks")
    
    # Print the number of trucks by freight forwarder
    print("\nTrucks by freight forwarder:")
    for ff in sorted(active_trucks['freight_forwarder'].unique()):
        ff_count = len(active_trucks[active_trucks['freight_forwarder'] == ff])
        print(f"  FF {ff}: {ff_count} trucks")
    
    # Plot the results
    save_path = 'output/figures/active_trucks_over_time.svg' if save_plot else None
    fig = plot_active_trucks(recording_counts, recording_counts_by_ff, 
                            first_last_dates, save_path)
    
    # Additional analysis - trucks by recording duration
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
    
    return active_trucks, recording_counts, recording_counts_by_ff, fig

# Call the function
if __name__ == "__main__":
    active_trucks, recording_counts, recording_counts_by_ff, fig = analyze_truck_activity()
    
    # If needed, you can also get a point-in-time count
    def count_active_trucks_at_time(active_trucks, target_time):
        """Count how many trucks were recording at a specific time"""
        if isinstance(target_time, str):
            target_time = pd.to_datetime(target_time, utc=True)
            
        active_count = len(active_trucks[
            (active_trucks['start_time'] <= target_time) & 
            (active_trucks['stop_time'] >= target_time)
        ])
        
        return active_count
    
    # Example usage for a specific time
    target_time = "2021-09-15 12:00:00"
    active_count = count_active_trucks_at_time(active_trucks, target_time)
    print(f"\nTrucks recording at {target_time}: {active_count}")
    
    # Count by freight forwarder at a specific time
    def count_active_trucks_by_ff_at_time(active_trucks, target_time):
        """Count how many trucks were recording at a specific time, broken down by freight forwarder"""
        if isinstance(target_time, str):
            target_time = pd.to_datetime(target_time, utc=True)
            
        # Filter to active trucks at target time
        active_at_time = active_trucks[
            (active_trucks['start_time'] <= target_time) & 
            (active_trucks['stop_time'] >= target_time)
        ]
        
        # Group and count by freight forwarder
        return active_at_time.groupby('freight_forwarder').size().reset_index(name='count')
    
    # Example usage for a specific time
    active_by_ff = count_active_trucks_by_ff_at_time(active_trucks, target_time)
    print(f"\nTrucks recording at {target_time} by freight forwarder:")
    
    # Print freight forwarders by number only
    for _, row in active_by_ff.iterrows():
        ff = row['freight_forwarder']
        print(f"  FF {ff}: {row['count']} trucks")