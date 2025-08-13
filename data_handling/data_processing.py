"""
Truck Fleet Data Processing Script

This script provides comprehensive data processing functions for truck fleet analysis,
including trip preprocessing, tour correction, stop analysis, and energy calculations.
It is a core component of the truck fleet electrification analysis workflow.

Version: 1.0
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import joblib

# Initialize joblib memory cache for function memoization
cache = joblib.Memory(".cache")

# Define location types for classification
LOCATIONS = ['home_base', 'rest_area', 'service_area_fuel', 'industrial_area', 'other_area']

def load_data():
    """
    Load trip and fleet data from CSV files.
    
    Returns
    -------
    tuple
        A tuple containing:
        - df_trips_unfiltered : pandas.DataFrame
            Raw trip data with track_id as index
        - df_fleet : pandas.DataFrame
            Fleet information with vehicle_id as index
    """
    df_trips_unfiltered = pd.read_csv(
        'input/stations/tracks.csv', 
        index_col='track_id', 
        parse_dates=['start_time', 'stop_time']
    )
    df_fleet = pd.read_csv('input/home/fleet.csv', index_col='vehicle_id')
    
    return df_trips_unfiltered, df_fleet


def load_speed_data(zf, i_veh, i_trip):
    """
    Load speed data for a specific vehicle and trip from zip file.
    
    Parameters
    ----------
    zf : zipfile.ZipFile
        Zip file containing speed data
    i_veh : int
        Vehicle identifier
    i_trip : int
        Trip identifier
        
    Returns
    -------
    pandas.DataFrame
        Speed data for the specified vehicle and trip
    """
    df_speed = pd.read_csv(zf.open(f'input/home/speed/{i_veh}/{i_trip}.csv'))
    return df_speed


def preprocess_trips_data(df_trips_unfiltered, df_fleet):
    """
    Preprocess and filter trip data for analysis.
    
    This function applies quality filters and data transformations:
    - Filters out trips shorter than 1 km
    - Removes trips with average speeds exceeding maximum speeds
    - Eliminates trips with negative durations
    - Converts units and adds derived columns
    
    Parameters
    ----------
    df_trips_unfiltered : pandas.DataFrame
        Raw trip data with track_id as index
    df_fleet : pandas.DataFrame
        Fleet information with vehicle_id as index
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed trip data with quality filters applied
    """
    # Create a copy to avoid modifying the original dataframe
    df_trips = df_trips_unfiltered.copy()
    
    # Apply quality filters
    df_trips = df_trips.loc[df_trips_unfiltered.distance > 1000]
    df_trips = df_trips.loc[df_trips_unfiltered.avg_speed <= df_trips_unfiltered.max_speed]
    
    # Convert distance to kilometers
    df_trips['distance_km'] = df_trips['distance'] / 1000
    
    # Convert time strings to datetime objects
    df_trips['stop_time'] = pd.to_datetime(df_trips['stop_time'], format='ISO8601')
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], format='ISO8601')
    
    # Calculate trip duration in seconds and hours
    df_trips['duration'] = (df_trips['stop_time'] - df_trips['start_time']) / pd.Timedelta(seconds=1)
    df_trips = df_trips[df_trips['duration'] > 0]
    df_trips['duration_h'] = df_trips['duration'] / 3600
    
    # Convert speeds to km/h
    df_trips['max_speed_kmh'] = df_trips['max_speed'] * 3.6
    df_trips['avg_speed_kmh'] = df_trips['avg_speed'] * 3.6
    
    # Merge fleet information
    fleet_columns = ['gross_vehicle_weight', 'total_mass_with_trailer', 'axle_class']
    df_trips = df_trips.merge(
        df_fleet[fleet_columns], 
        left_on='vehicle_id', 
        right_index=True, 
        how='left'
    )
    
    # Create other_area classification
    df_trips['other_area'] = (
        (~df_trips['home_base']) & 
        (~df_trips['service_area_fuel']) & 
        (~df_trips['rest_area']) & 
        (~df_trips['industrial_area'])
    )
    
    return df_trips


def fix_faulty_tour_assignment(trips, trips_unfiltered):
    """
    Correct tracks that have been incorrectly assigned to tours.
    
    This function addresses the issue where tracks are assigned to the wrong tour
    when the previous tour doesn't end at a home base. Subsequent tracks are then
    incorrectly added to the tour until a home base is reached, even if they belong
    to different tours.
    
    Tracks with incorrect tour assignments are given new synthetic tour IDs
    in the format x00000 for clear identification.
    
    Parameters
    ----------
    trips : pandas.DataFrame
        Preprocessed trip data
    trips_unfiltered : pandas.DataFrame
        Raw unfiltered trip data for reference
        
    Returns
    -------
    pandas.DataFrame
        Trip data with corrected tour assignments
    """
    df_trips = trips.copy()
    df_trips_unfiltered = trips_unfiltered.copy()

    # Identify tours with multiple vehicle IDs (problematic tours)
    tours_with_multiple_vehicles = df_trips.groupby('tour_id')['vehicle_id'].nunique()
    problematic_tours = tours_with_multiple_vehicles[tours_with_multiple_vehicles > 1].index.tolist()

    if problematic_tours:
        print(f"Found {len(problematic_tours)} tours with multiple vehicle IDs")
        synthetic_tour_id_counter = 0
        
        for tour_id in problematic_tours:
            # Get all tracks for this tour
            tour_tracks = df_trips[df_trips['tour_id'] == tour_id].sort_values('track_id')
            
            # Get unique vehicle IDs in this tour
            unique_vehicles = tour_tracks['vehicle_id'].unique()
            
            # Keep the first vehicle ID as the original tour
            first_vehicle = tour_tracks['vehicle_id'].iloc[0]
            
            # Assign new tour IDs to tracks with different vehicle IDs
            for vehicle_id in unique_vehicles:
                if vehicle_id != first_vehicle:
                    # Create a new synthetic tour ID
                    new_tour_id = 100000 * (synthetic_tour_id_counter + 1)
                    synthetic_tour_id_counter += 1
                    
                    # Get indices of tracks with this vehicle ID in the tour
                    vehicle_track_indices = tour_tracks[tour_tracks['vehicle_id'] == vehicle_id].index
                    
                    # Assign the new tour ID
                    df_trips.loc[vehicle_track_indices, 'tour_id'] = new_tour_id
                    print(f"Split tour {tour_id}: assigned new tour_id {new_tour_id} to vehicle {vehicle_id}")

    # Update last track information from unfiltered data
    df_last_tracks = df_trips_unfiltered.reset_index().sort_values(['tour_id', 'stop_time']).groupby('tour_id').last()
    last_track_mapping = df_last_tracks[['track_id', 'cid']].reset_index()

    # Update cid and home_base for last tracks of each tour
    for _, row in last_track_mapping.iterrows():
        tour = row['tour_id']
        if tour in df_trips['tour_id'].values:  # Check that the tour hasn't been filtered out
            last_track_id = df_trips.loc[df_trips['tour_id'] == tour].index.max()
            df_trips.loc[last_track_id, 'cid'] = row['cid']
            df_trips.loc[last_track_id, 'home_base'] = True

    return df_trips


def fix_faulty_tour_endings(trips, trips_unfiltered):
    """
    Fix tours that don't correctly end at home bases.
    
    This function iterates through each freight forwarder and corrects tours
    where the last trip is not properly marked as returning to a home base.
    It assigns appropriate home base CIDs based on the freight forwarder's
    known home base locations.
    
    Parameters
    ----------
    trips : pandas.DataFrame
        Preprocessed trip data
    trips_unfiltered : pandas.DataFrame
        Raw unfiltered trip data for reference
        
    Returns
    -------
    pandas.DataFrame
        Trip data with corrected tour endings
    """
    # Create copies to avoid modifying the original dataframes
    df = trips.copy()
    df_raw = trips_unfiltered.copy()
    
    # Get unique freight forwarders
    freight_forwarders = df['freight_forwarder'].unique()
    
    # Track changes for reporting
    total_fixed_tours = 0
    
    # Define home base locations for each freight forwarder
    homebase_map = {
        1: [0], 
        2: [745], 
        3: [2, 4, 20, 26], 
        4: [502], 
        5: [102, 868, 1251, 1265], 
        6: [412, 809, 842]
    }

    # Process each freight forwarder separately
    for ff in freight_forwarders:
        # Get data for this freight forwarder
        ff_data = df[df['freight_forwarder'] == ff]
        bases = homebase_map[ff]
        
        # Find the most common base for this freight forwarder
        base_counts = {}
        for base in bases:
            count = len(ff_data[ff_data['cid'] == base])
            base_counts[base] = count
            
        # Find the most common base
        most_common_base = max(base_counts, key=base_counts.get) if base_counts else bases[0]

        # Find the last track of each tour
        last_tracks = ff_data.sort_values(['tour_id', 'stop_time']).groupby('tour_id').last()
        
        # Identify tours that need fixing (don't end at home base)
        tours_to_fix = last_tracks[~last_tracks['cid'].isin(bases)].index.tolist()
        fixed_count = len(tours_to_fix)
        total_fixed_tours += fixed_count
        
        # Fix tours that don't end at home base
        if fixed_count > 0:
            for tour_id in tours_to_fix:
                # Get the last track for this tour
                tour_tracks = df[(df['tour_id'] == tour_id)].sort_values('stop_time')
                last_track_id = tour_tracks.index[-1]
                tour_tracks_raw = df_raw[(df_raw['tour_id'] == tour_id)].sort_values('stop_time')

                # Find matching home bases in the tour
                matching_bases = [cid for cid in tour_tracks_raw['cid'] if cid in bases]
            
                if matching_bases:
                    # Use the first matching base found in the tour
                    df.loc[last_track_id, 'cid'] = matching_bases[0]
                    print(f'Fixing tour {tour_id}: assigning last track to cid {matching_bases[0]}')
                else: 
                    # Assign to the most common home base if no matches found
                    df.loc[last_track_id, 'cid'] = most_common_base 
                    print(f'Fixing tour {tour_id}: assigning last track to most common home base cid {most_common_base}')
                
            print(f"Fixed {fixed_count} tours for freight forwarder {ff}")
    
    print(f"Total fixed tours: {total_fixed_tours}")
    return df

@cache.cache
def process_stops_data(df_trips):
    """
    Process trip data to create stop analysis with location classification and rest times.
    
    This function creates mutually exclusive location classifications and calculates
    rest times between consecutive trips for each vehicle.
    
    Parameters
    ----------
    df_trips : pandas.DataFrame
        Preprocessed trip data with location flags
        
    Returns
    -------
    pandas.DataFrame
        Stop data with location classification and rest time calculations
    """
    df_stops = df_trips.copy()
    
    # Create mutually exclusive location classifications
    # Priority order: home_base > service_area_fuel > rest_area > industrial_area > other_area
    df_stops['service_area_fuel'] = (~df_trips['home_base']) & df_trips['service_area_fuel']
    df_stops['rest_area'] = (~df_trips['home_base']) & (~df_trips['service_area_fuel']) & df_trips['rest_area']
    df_stops['industrial_area'] = (
        (~df_trips['home_base']) & 
        (~df_trips['service_area_fuel']) & 
        (~df_stops['rest_area']) & 
        df_trips['industrial_area']
    )
    df_stops['other_area'] = (
        (~df_trips['home_base']) & 
        (~df_trips['service_area_fuel']) & 
        (~df_stops['rest_area']) & 
        (~df_stops['industrial_area'])
    )
    
    # Create location label from boolean flags
    df_stops['location'] = df_stops[LOCATIONS].idxmax(axis=1).str.replace('_', ' ')
    
    # Calculate rest time between consecutive trips for each vehicle
    rest_time = df_trips.groupby('vehicle_id').start_time.shift(-1) - df_trips['stop_time']
    rest_time = rest_time.fillna(pd.Timedelta(seconds=0))  # Set rest time to 0 for last trip
    df_stops = df_stops.assign(rest_time=rest_time / pd.Timedelta(seconds=1))
    
    # Convert rest time to hours
    df_stops = df_stops.assign(rest_time_h=df_stops['rest_time'] / 3600)
    
    return df_stops

def time_to_seconds(t):
    """
    Convert time object to total seconds since midnight.
    
    Parameters
    ----------
    t : datetime.time or pandas.Timestamp
        Time object to convert
        
    Returns
    -------
    int
        Total seconds since midnight
    """
    return t.hour * 3600 + t.minute * 60 + t.second


def seconds_to_time(s):
    """
    Convert seconds since midnight to time object.
    
    Parameters
    ----------
    s : int or float
        Seconds since midnight
        
    Returns
    -------
    pandas.Timestamp
        Time object representing the given seconds
    """
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(s, unit='s')


# =============================================================================
#                              META DATA CALCULATION
# =============================================================================


def calculate_meta_data(df_trips_unfiltered, df_trips, df_fleet):
    """
    Calculate comprehensive metadata statistics for the truck fleet dataset.
    
    This function computes various metrics including distance, time, speed,
    location distribution, and temporal patterns for both overall fleet
    and individual freight forwarders.
    
    Parameters
    ----------
    df_trips_unfiltered : pandas.DataFrame
        Raw unfiltered trip data for total recording counts
    df_trips : pandas.DataFrame
        Preprocessed trip data for analysis
    df_fleet : pandas.DataFrame
        Fleet information with vehicle specifications
        
    Returns
    -------
    dict
        Dictionary containing all calculated metadata metrics
    """
    # Ensure datetime format for time-based calculations
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['stop_time'] = pd.to_datetime(df_trips['stop_time'], utc=True)
    
    # Basic fleet and trip statistics
    total_distance = df_trips['distance'].sum()
    total_time = df_trips['duration'].sum()
    trips_longer_1km = len(df_trips)
    tours = len(df_trips[['vehicle_id', 'tour_id']].drop_duplicates())
    fleet_size = len(df_fleet)
    median_vehicle_driving_distance_km = df_trips.groupby('vehicle_id')['distance'].sum().median() / 1000
    
    # Speed and signal quality metrics
    avg_speed_per_trip = df_trips['avg_speed'].mean()
    max_speed = df_trips['max_speed'].max()
    total_signal_loss = df_trips['n_signal_loss'].sum()
    avg_signal_loss_ratio = df_trips['r_signal_loss'].mean()
    
    # GPS quality metrics (if available)
    avg_hdop = df_trips['avg_hdop'].mean() if 'avg_hdop' in df_trips.columns else None
    max_hdop = df_trips['max_hdop'].max() if 'max_hdop' in df_trips.columns else None
    
    # Statistical summaries
    rest_time_stats = df_trips['duration_h'].describe()
    distance_stats = df_trips['distance_km'].describe()
    
    # Location distribution analysis
    location_distribution = df_trips[LOCATIONS].mean() * 100
    
    # Freight forwarder specific metrics
    ff_metrics = df_trips.groupby('freight_forwarder').agg({
        'distance': 'sum',
        'duration': 'sum',
        'vehicle_id': 'nunique'
    })
    ff_metrics.columns = ['total_distance', 'total_duration', 'num_vehicles']
    ff_metrics['total_distance_km'] = ff_metrics['total_distance'] / 1000
    ff_metrics['total_duration_h'] = ff_metrics['total_duration'] / 3600
    
    # Spatial metrics (placeholders for future implementation)
    max_latitude = 0  # Placeholder for maximum latitude
    min_latitude = 0  # Placeholder for minimum latitude
    max_longitude = 0  # Placeholder for maximum longitude
    min_longitude = 0  # Placeholder for minimum longitude
    
    # Data quality metrics
    avg_points_per_trip = df_trips['track_gap'].mean()
    
    # Daily aggregation metrics
    avg_trip_time_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['duration'].mean().mean()
    max_trip_time_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['duration'].sum().max()
    
    avg_trip_distance_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['distance'].mean().mean()
    max_trip_distance_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['distance'].sum().max()
    
    # Tour-level metrics
    avg_trips_per_tour = df_trips.groupby('tour_id').size().mean()
    avg_signal_loss_per_tour = df_trips.groupby('tour_id')['r_signal_loss'].mean().mean()
    
    # Recording statistics
    total_recordings = len(df_trips_unfiltered)
    recordings_under_1km = len(df_trips_unfiltered[df_trips_unfiltered['distance'] < 1000])  
    
    # Time range
    min_start_time = df_trips['start_time'].min()
    max_start_time = df_trips['start_time'].max()
    
    # Temporal patterns per truck
    avg_trips_per_day_per_truck = df_trips.groupby('vehicle_id').size().mean()
    avg_trip_duration_h = df_trips['duration_h'].mean()
    avg_trip_distance_km = df_trips['distance_km'].mean()
    
    # Daily distance calculations
    df_trips['date'] = df_trips['start_time'].dt.date
    daily_distances = df_trips.groupby(['vehicle_id', 'date'])['distance_km'].sum().reset_index()
    avg_daily_distance_per_truck_km = daily_distances.groupby('vehicle_id')['distance_km'].mean().mean()

    # Speed conversion to km/h
    avg_speed_per_trip_kmh = avg_speed_per_trip * 3.6

    # Calculate average daily start and end times
    first_trips = df_trips.groupby(['vehicle_id', 'date'])['start_time'].min().reset_index()
    last_trips = df_trips.groupby(['vehicle_id', 'date'])['stop_time'].max().reset_index()

    avg_daily_start_time_seconds = (
        first_trips['start_time'].dt.hour.mean() * 3600 + 
        first_trips['start_time'].dt.minute.mean() * 60 + 
        first_trips['start_time'].dt.second.mean()
    )
    avg_daily_end_time_seconds = (
        last_trips['stop_time'].dt.hour.mean() * 3600 + 
        last_trips['stop_time'].dt.minute.mean() * 60 + 
        last_trips['stop_time'].dt.second.mean()
    )
    avg_daily_start_time = seconds_to_time(avg_daily_start_time_seconds).strftime('%H:%M:%S')
    avg_daily_end_time = seconds_to_time(avg_daily_end_time_seconds).strftime('%H:%M:%S')

    meta_data = {
        'total_distance': total_distance,
        'total_time': total_time,
        'trips': trips_longer_1km,
        'tours': tours,
        'fleet_size': fleet_size,
        'median_vehicle_distance': median_vehicle_driving_distance_km,
        'avg_speed_per_trip': avg_speed_per_trip,
        'max_speed': max_speed,
        'total_signal_loss': total_signal_loss,
        'avg_signal_loss_ratio': avg_signal_loss_ratio,
        'avg_hdop': avg_hdop,
        'max_hdop': max_hdop,
        'rest_time_stats': rest_time_stats,
        'distance_stats': distance_stats,
        'location_distribution': location_distribution,
        'max_latitude': max_latitude,
        'min_latitude': min_latitude,
        'max_longitude': max_longitude,
        'min_longitude': min_longitude,
        'avg_points_per_trip': avg_points_per_trip,
        'avg_trip_time_per_day': avg_trip_time_per_day,
        'max_trip_time_per_day': max_trip_time_per_day,
        'avg_trip_distance_per_day': avg_trip_distance_per_day,
        'max_trip_distance_per_day': max_trip_distance_per_day,
        'avg_trips_per_tour': avg_trips_per_tour,
        'avg_signal_loss_per_tour': avg_signal_loss_per_tour,
        'share_trips_shorter_5h': len(df_trips.loc[df_trips.duration_h <= 5]) / len(df_trips),
        'trips_shorter_1000m': recordings_under_1km,
        'total_recordings': total_recordings,
        'min_start_time': min_start_time,
        'max_start_time': max_start_time,
        # Temporal metrics
        'avg_trips_per_day_per_truck': avg_trips_per_day_per_truck,
        'avg_trip_duration_h': avg_trip_duration_h,
        'avg_trip_distance_km': avg_trip_distance_km,
                'avg_daily_distance_per_truck_km': avg_daily_distance_per_truck_km,
                'avg_daily_start_time': avg_daily_start_time,
        'avg_daily_end_time': avg_daily_end_time,
'avg_speed_per_trip_kmh': avg_speed_per_trip_kmh,
        # Freight forwarder metrics
        'ff_metrics': ff_metrics
    }
    return meta_data

def meta_data_to_df(meta_data):
    general_data = {
        'Metric': [
            'Recorded distance / km', 'Recorded time / h', 'Trips', 'Tours', 'Vehicles', 
            'Median vehicle distance / km', 'Share of trips shorter than 5h', 'Trips shorter than 1 km', 'Total recordings', 
            'Min start time', 'Max start time'
        ],
        'Value': [
            round(meta_data['total_distance'] / 1000, 3),
            round(meta_data['total_time'] / 3600, 3),
            meta_data['trips'],
            meta_data['tours'],
            meta_data['fleet_size'],
            round(meta_data['median_vehicle_distance'], 3),
            round(meta_data['share_trips_shorter_5h'], 3),
            meta_data['trips_shorter_1000m'],
            meta_data['total_recordings'],
            meta_data['min_start_time'],
            meta_data['max_start_time']
        ]
    }

    temporal_data = {
        'Metric': [
            'Average Trips per Day per Truck', 'Average Trip Duration / h', 
            'Average Trip Distance / km', 'Average Trips per Tour', 
            'Average Daily Distance per Truck / km', 'Average Daily Start Time', 
            'Average Daily End Time', 'Average Speed per Trip/ km/h'
        ],
        'Value': [
            round(meta_data['avg_trips_per_day_per_truck'], 3),
            round(meta_data['avg_trip_duration_h'], 3),
            round(meta_data['avg_trip_distance_km'], 3),
            round(meta_data['avg_trips_per_tour'], 3),
            round(meta_data['avg_daily_distance_per_truck_km'], 3),
            meta_data['avg_daily_start_time'],
            meta_data['avg_daily_end_time'],
            round(meta_data['avg_speed_per_trip_kmh'], 3)
        ]
    }

    spatial_data = {
        'Metric': ['Maximum Latitude', 'Minimum Latitude', 'Maximum Longitude', 'Minimum Longitude'],
        'Value': [
            meta_data['max_latitude'],
            meta_data['min_latitude'],
            meta_data['max_longitude'],
            meta_data['min_longitude']
        ]
    }
    
    # Format the freight forwarder metrics dataframe
    ff_data = meta_data['ff_metrics'].reset_index()
    ff_data = ff_data.rename(columns={
        'freight_forwarder': 'Fleet',
        'num_vehicles': 'Vehicles',
        'total_distance_km': 'Recorded distance / 1000 km',
        'total_duration_h': 'Recorded time / h'
    })
    ff_data = ff_data[['Fleet', 'Vehicles', 'Recorded distance / 1000 km', 'Recorded time / h']]
    # Round the numeric columns
    ff_data['Recorded distance / 1000 km'] = ff_data['Recorded distance / 1000 km'].round(2)
    ff_data['Recorded time / h'] = ff_data['Recorded time / h'].round(2)

    general_df = pd.DataFrame(general_data)
    temporal_df = pd.DataFrame(temporal_data)
    spatial_df = pd.DataFrame(spatial_data)
    
    return general_df, temporal_df, spatial_df, ff_data



def calculate_weekly_distances(df_trips):
    trips = df_trips.copy()
    trips['start_time'] = pd.to_datetime(trips['start_time'], utc=True)
    trips['start_time'] = trips['start_time'].dt.tz_localize(None)
    trips['stop_time'] = trips['stop_time'].dt.tz_localize(None)
    trips['start_time_7d'] = trips['start_time'].dt.to_period("W-SUN")
    distances_km = trips.groupby(['start_time_7d'])['distance'].sum() / 1000
    return distances_km


 # speed converted to kilometers per hour for visualization
def convert_speed_to_kmh(df_speed):
    df_speed['kmh'] = df_speed['speed'] * 3.6

    # relative time in seconds since beginning of trip
    df_speed['rel_time'] = df_speed.epoch - df_speed.epoch.min()
    df_speed.set_index('rel_time', inplace=True)

    return df_speed


# def calculate_occupation(df_trips):
#     # ! Output is not actually used anywhere !
#     # TODO delete?? Not used and i think it is not correct
#     # Creates 2x the amount of rows as the original df_trips (as it adds a new occupation between each trip (occupation = driving))

#     df_occupation = df_trips[['vehicle_id', 'start_time', 'stop_time']].copy()
#     df_occupation.rename(columns={'start_time': 'stop_time', 'stop_time': 'start_time'}, inplace=True)
    
#     for loc in [*LOCATIONS, 'other_area']:
#         if loc not in df_trips.columns:
#             df_trips[loc] = False

#     df_occupation = df_occupation.assign(occupation=df_trips[[*LOCATIONS, 'other_area']].idxmax(axis=1))

#     df_driving_occupation = df_trips[['vehicle_id', 'start_time', 'stop_time']].copy()
#     df_driving_occupation['occupation'] = 'driving'

#     df_occupation = pd.concat([df_occupation, df_driving_occupation.reset_index(drop=True)])
#     #df_occupation = df_occupation.append(df_driving_occupation.reset_index(drop=True))

#     df_occupation['start_time'] = pd.to_datetime(df_occupation['start_time'], utc=True)
#     df_occupation['stop_time'] = pd.to_datetime(df_occupation['stop_time'], utc=True)

#     df_occupation['duration'] = df_occupation.stop_time - df_occupation.start_time
#     df_occupation['duration'] = df_occupation.duration.dt.total_seconds() / 3600
    
#     df_occupation.set_index('start_time', inplace=True)
#     df_occupation = df_occupation.tz_convert('Europe/Berlin') # This does only converts the timezone of the index (start_time)!!

#     return df_occupation


# def prepare_occupation_data(df_occupation):
#     # out of date?? (No used in the original repo)
#     resampled = df_occupation.groupby(['vehicle_id']).resample('1min').ffill()[['occupation', 'duration']]
#     resampled = resampled.reset_index()

#     resampled['dow'] = resampled.start_time.dt.dayofweek
#     resampled['hour'] = resampled.start_time.dt.hour
#     resampled = resampled.loc[resampled.dow < 6] # only week days
#     resampled = resampled.loc[resampled.duration < 24] # remove very long stays

#     truck_day = resampled.groupby(['occupation', 'hour']).occupation.count()
#     truck_day = truck_day.unstack(level=-1, fill_value=0)
#     return truck_day


def resample_occupation_data(df_stops, df_trips):
    df_occupation = df_stops[['vehicle_id', 'start_time', 'stop_time']].copy()
    df_occupation.rename(columns={'start_time': 'stop_time', 'stop_time': 'start_time'}, inplace=True)
    df_occupation = df_occupation.assign(occupation=df_stops[[*LOCATIONS, 'other_area']].idxmax(axis=1))

    df_driving_occupation = df_trips[['vehicle_id', 'start_time', 'stop_time']].copy()
    df_driving_occupation['occupation'] = 'driving'

    #df_occupation = df_occupation.append(df_driving_occupation.reset_index(drop=True))
    df_occupation = pd.concat([df_occupation, df_driving_occupation.reset_index(drop=True)], ignore_index=True)

    df_occupation['start_time'] = pd.to_datetime(df_occupation['start_time'], utc=True)
    df_occupation['stop_time'] = pd.to_datetime(df_occupation['stop_time'], utc=True)

    df_occupation['duration'] = df_occupation.stop_time - df_occupation.start_time
    df_occupation['duration'] = df_occupation.duration.dt.total_seconds() / 3600
    df_occupation.set_index('start_time', inplace=True)
    df_occupation = df_occupation.tz_convert('Europe/Berlin')

    resampled = df_occupation.groupby(['vehicle_id']).resample('1min').ffill()[['occupation', 'duration']]
    resampled = resampled.reset_index()

    resampled['dow'] = resampled.start_time.dt.dayofweek
    resampled['hour'] = resampled.start_time.dt.hour

    resampled = resampled.loc[resampled.dow < 6]
    resampled = resampled.loc[resampled.duration < 24]

    truck_day = resampled.groupby(['occupation', 'hour']).occupation.count()
    truck_day = truck_day.unstack(level=-1, fill_value=0)

    return truck_day


def calculate_rest_times(df_stops):
    df_rt_joined = pd.DataFrame(index=df_stops['vehicle_id'].unique())
    for flag in [*LOCATIONS, 'other_area']:
        rest_times = df_stops[df_stops[flag]].groupby('vehicle_id')['rest_time'].sum()
        df_rt_joined[f'rest_time_{flag}'] = rest_times
    
    df_rt_joined.rename(columns={f'rest_time_{l}': l for l in [*LOCATIONS, 'other_area']}, inplace=True)
    return df_rt_joined


def calculate_rest_times_and_driving(df_stops, df_trips):
    """
    Calculate the time spend in each occupation and the time spend driving for each vehicle in seconds"""
    df_rt_joined = pd.DataFrame(index=df_trips['vehicle_id'].unique())

    for flag in [*LOCATIONS, 'other_area']:
        rest_times = df_stops[df_stops[flag]].groupby('vehicle_id')['rest_time'].sum()
        df_rt_joined[f'rest_time_{flag}'] = rest_times

    df_rt_joined.rename(columns={f'rest_time_{l}': l for l in [*LOCATIONS, 'other_area']}, inplace=True)

    df_driving = df_trips.groupby('vehicle_id')['duration'].sum()
    
    # Get the freight forwarder for each vehicle
    freight_forwarder = df_trips.groupby('vehicle_id')['freight_forwarder'].first()

    # Join driving time and freight forwarder to the rest times dataframe
    df_rt_joined_plot = df_rt_joined.join(df_driving).rename(columns={'duration': 'driving'})
    df_rt_joined_plot = df_rt_joined_plot.join(freight_forwarder)

    return df_rt_joined_plot


def aggregate_driving_times(df_trips, df_rt_joined):
    df_driving = df_trips.groupby('vehicle_id')['duration'].sum()
    df_rt_joined_plot = df_rt_joined.join(df_driving).rename(columns={'duration': 'driving'})
    return df_rt_joined_plot


def calculate_drive_pause_by_weekday(df_trips):
    # Convert 'start_time' and 'stop_time' into datetime columns and set timezone to UTC
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['stop_time'] = pd.to_datetime(df_trips['stop_time'], utc=True)

    # Calculate the duration of each trip in hours
    df_trips['duration_h'] = (df_trips['stop_time'] - df_trips['start_time']).dt.total_seconds() / 3600

    # Add a new column for the weekday
    df_trips['weekday'] = df_trips['start_time'].dt.weekday  # 0 = Monday, 6 = Sunday

    # Calculate total drive time and pause time for each vehicle per weekday
    drive_times = df_trips.groupby(['vehicle_id', 'weekday'])['duration_h'].sum().reset_index(name='drive_time')
    pause_times = df_trips.groupby(['vehicle_id', 'weekday'])['track_gap'].sum().reset_index(name='pause_time')

    # Calculate remaining pause times (beginning and end of the period)
    first_trip_times = df_trips.groupby(['vehicle_id', 'weekday'])['start_time'].min().reset_index()
    last_trip_times = df_trips.groupby(['vehicle_id', 'weekday'])['stop_time'].max().reset_index()
    
    # Calculate the total duration of the dataset period
    dataset_start_time = df_trips['start_time'].min()
    dataset_end_time = df_trips['stop_time'].max()
    total_period_duration = (dataset_end_time - dataset_start_time).total_seconds() / 3600

    # Calculate remaining pause times
    first_trip_times['remaining_pause_at_start'] = (first_trip_times['start_time'] - dataset_start_time).dt.total_seconds() / 3600
    last_trip_times['remaining_pause_at_end'] = (dataset_end_time - last_trip_times['stop_time']).dt.total_seconds() / 3600

    # Merge dataframes
    merged_times = pd.merge(drive_times, pause_times, on=['vehicle_id', 'weekday'])
    merged_times = pd.merge(merged_times, first_trip_times[['vehicle_id', 'weekday', 'remaining_pause_at_start']], on=['vehicle_id', 'weekday'])
    merged_times = pd.merge(merged_times, last_trip_times[['vehicle_id', 'weekday', 'remaining_pause_at_end']], on=['vehicle_id', 'weekday'])

    # Calculate average times per weekday
    avg_times = merged_times.groupby('weekday').mean().reset_index()

    return avg_times


def aggregate_hourly_distance(df_trips, per_day=False):
    df_trips_copy = df_trips.copy()

    # Convert time columns to datetime
    df_trips_copy['start_time'] = pd.to_datetime(df_trips_copy['start_time'], utc=True)
    df_trips_copy['stop_time'] = pd.to_datetime(df_trips_copy['stop_time'], utc=True)

    # Set the index to stop_time and convert to the desired timezone
    df_trips_copy.set_index('stop_time', inplace=True)
    df_trips_copy = df_trips_copy.tz_convert('Europe/Berlin')

    # Resample to hourly frequency and calculate the distance driven by trucks arriving in the last hour
    resampled = df_trips_copy.resample('1H').agg({
        'distance_km': 'sum',
        'vehicle_id': 'nunique'
    }).rename(columns={'distance_km': 'absolute_distance', 'vehicle_id': 'num_trucks'})

    # Calculate the number of days in the recording
    num_days = (df_trips_copy.index.max() - df_trips_copy.index.min()).days + 1

    # Calculate the distance driven by trucks arriving in the last hour per day
    resampled['distance_per_day'] = resampled['absolute_distance'] / num_days

    if per_day:
        # Create a DataFrame with a row per day
        resampled['date'] = resampled.index.date
        daily_result = resampled.pivot_table(index='date', columns=resampled.index.hour + 1, values='absolute_distance', fill_value=0)
        daily_result.insert(0, 0, 0)
        return daily_result

    # Create the resulting DataFrame with a single row
    result = pd.DataFrame(index=['num_days', 'absolute_distance', 'distance_per_day', 'num_trucks', 'avg_num_trucks_per_day'])
    for hour in range(24):
        result[hour + 1] = [
            num_days,
            resampled.loc[resampled.index.hour == hour, 'absolute_distance'].sum(),
            resampled.loc[resampled.index.hour == hour, 'distance_per_day'].sum(),
            resampled.loc[resampled.index.hour == hour, 'num_trucks'].sum(),
            resampled.loc[resampled.index.hour == hour, 'num_trucks'].sum() / num_days
        ]

    # insert 0th column for easier reading
    result.insert(0, 0, [num_days, 0, 0, 0, 0])
    return result


@cache.cache
def aggregate_tours(df_trips, energy=False):

    if energy:
        df_tours = df_trips.groupby('tour_id').agg({
            'distance': 'sum',
            'distance_km': 'sum',
            'duration': 'sum',
            'duration_h': 'sum',
            'start_time': 'min',
            'stop_time': 'max',
            'freight_forwarder': 'min',
            'energy_consumption_kwh_cleaned': 'sum',
            'energy_recharged_kwh': 'sum',
            #'energy_recharged_kwh_potential': 'sum',
            'battery_energy_kwh': ['min', 'mean'],
            'soc': ['min', 'mean'],
            'soc_no_public_charging': ['min', 'mean'],
            'track_id': 'count',
            'cid': 'last',
            }
        ).reset_index()

        df_tours.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_tours.columns]

        # Rename soc_start to soc_arrival
        df_tours = df_tours.rename(columns={
            'stop_time_max': 'end_time',
        })
        
        # Drop suffixes for all columns except battery_energy and soc
        df_tours.columns = [
            col.replace('_min', '').replace('_max', '').replace('_mean', '').replace('_sum', '').replace('_last', '')
            if not col.startswith('battery_energy_kwh') and not col.startswith('soc') 
            else col 
            for col in df_tours.columns
        ]

        # Create tours_driving dataframe that only considers activities driving activities 
        # (i.e. activities with a track_id)
        tours_driving = df_trips[df_trips['track_id'].notna()].groupby('tour_id').agg({
            'duration': 'sum',
            'duration_h': 'sum',
            'stop_time': 'max',
        }).reset_index()

        tours_driving = tours_driving.rename(columns={
            'duration': 'driving_duration',
            'duration_h': 'driving_duration_h',
        })

        df_tours = df_tours.merge(tours_driving, on='tour_id', how='left')

    else:
        df_tours = df_trips.groupby('tour_id').agg({
            'distance': 'sum',
            'distance_km': 'sum',
            'duration': 'sum',
            'duration_h': 'sum',
            'start_time': 'min',
            'cid': 'last',
            **{col: 'max' for col in df_trips.columns if col not in ['distance', 'distance_km', 'duration', 'duration_h', 'start_time', 'tour_id', 'track_id', 'cid']}
        }).reset_index()

    return df_tours


def daily_energy_demands(tours_df, threshold, charging_powers):
    """
    Aggregates the total energy consumption by cid and day from the tours data.
    
    Parameters:
    -----------
    tours_df : pandas.DataFrame
        DataFrame containing tours data with columns 'cid', 'stop_time', 
        'energy_consumption_kwh_cleaned', and 'freight_forwarder'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'day', 'energy_demand_kwh', 'cid', and 'freight_forwarder'
    """
    # Make a copy to avoid modifying the original dataframe
    df = tours_df.copy()
    charging_power = charging_powers['home base']
    
    # Convert stop_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['stop_time']):
        df['stop_time'] = pd.to_datetime(df['stop_time'], utc=True)
    
    # Set stop time as index and convert it to local time
    df.set_index('stop_time', inplace=True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df.reset_index(inplace=True)
    # Extract the date (day) from stop_time
    df['day'] = df['stop_time'].dt.date
    
    """
    Correctly extracts the date considering local time. E.g:
    df.stop_time.dt.date.iloc[1084]
        datetime.date(2021, 11, 22)

    df.iloc[1084].stop_time
        Timestamp('2021-11-22 00:12:43.132000+0100', tz='Europe/Berlin')    

    df.iloc[1084].stop_time.tz_convert('UTC')
        Timestamp('2021-11-21 23:12:43.132000+0000', tz='UTC')
    """

    df['energy_demand_kwh'] = df['energy_consumption_kwh_cleaned'] - df['energy_recharged_kwh']
    df['energy_demand_kwh'] = df['energy_demand_kwh'].clip(upper=threshold)  # limit max energy demand to battery capacity

    # Group by day, cid, and freight_forwarder, calculating the difference between consumed and recharged energy
    daily_energy = df.groupby(['day', 'cid', 'freight_forwarder']).agg(
        energy_demand_kwh=('energy_demand_kwh', 'sum'),
    ).reset_index()
    
    # Reorder columns to match the requested order
    daily_energy = daily_energy[['day', 'energy_demand_kwh', 'cid', 'freight_forwarder']]

    directory = f"output/charging_loads/{charging_power}_kW"
    os.makedirs(directory, exist_ok=True)
    # Save the load profile
    # daily_energy.to_csv(f"output/charging_loads/{charging_power}_kW/daily_loads.csv", index=False)
    
    return daily_energy


def create_minute_occupation_dataframes(activities_file):
    """
    Creates dataframes showing minute-by-minute truck occupations per freight forwarder.
    Also adds a column showing how many trucks are logging data at each minute.
    
    Parameters:
    -----------
    activities_file : str
        Path to the activities CSV file containing truck occupation data
    
    Returns:
    --------
    dict
        Dictionary with freight forwarders as keys and dataframes as values
    """
    # Load the activities data
    df_activities = pd.read_csv(activities_file)
    
    # do everything in UTC and then convert the index of final df to local time
    df_activities['start_time'] = pd.to_datetime(df_activities['start_time'], utc=True, format='ISO8601')
    df_activities['stop_time'] = pd.to_datetime(df_activities['stop_time'], utc=True, format='ISO8601')
    
    # Round start time up to the next minute and stop time down to the previous minute 
    df_activities['start_time_rounded'] = df_activities['start_time'].apply(
        lambda x: x - timedelta(seconds=x.second, microseconds=x.microsecond) + timedelta(minutes=1)
    )
    df_activities['stop_time_rounded'] = df_activities['stop_time'].apply(
        lambda x: x - timedelta(seconds=x.second, microseconds=x.microsecond) 
    )
    
    # Get all unique occupations and freight forwarders
    occupations = df_activities['occupation'].unique()
    freight_forwarders = df_activities['freight_forwarder'].unique()
    
    # Create a dictionary to store the results
    result_dataframes = {}
    
    # Process each freight forwarder
    for ff in freight_forwarders:
        # Filter activities for this freight forwarder
        ff_activities = df_activities[df_activities['freight_forwarder'] == ff]

        # Get time range of recording of the respective freight forwarder 
        min_time = ff_activities['start_time_rounded'].min()
        max_time = ff_activities['stop_time_rounded'].max()
        
        # Get unique vehicles for this freight forwarder
        vehicles = ff_activities['vehicle_id'].unique()
        num_vehicles = len(vehicles)
        
        # Create a time index from min_time to max_time with minute frequency
        time_index = pd.date_range(start=min_time, end=max_time, freq='1T')
        
        # Create an empty dataframe with time as index and occupations as columns
        columns_list = list(occupations) + ['trucks_logging']  # Add trucks_logging column
        df_result = pd.DataFrame(index=time_index, columns=columns_list)
        df_result = df_result.fillna(0)  # Initialize with zeros
        
        # Find first and last logging time for each vehicle
        vehicle_logging_periods = {}
        for vehicle_id in vehicles:
            vehicle_data = ff_activities[ff_activities['vehicle_id'] == vehicle_id]
            first_log = vehicle_data['start_time_rounded'].min()
            last_log = vehicle_data['stop_time_rounded'].max()
            vehicle_logging_periods[vehicle_id] = (first_log, last_log)
        
        # For each activity, mark the truck's occupation for each minute
        for _, activity in ff_activities.iterrows():
            if activity['start_time_rounded'] <= activity['stop_time_rounded']:  # Make sure start is before stop after rounding
                # Get the time range for this activity
                activity_range = pd.date_range(
                    start=activity['start_time_rounded'], 
                    end=activity['stop_time_rounded'], 
                    freq='1T'
                )
                
                # Increment the count for this occupation during this time range
                occupation = activity['occupation']
                df_result.loc[activity_range, occupation] += 1
        
        # Count trucks logging data at each minute
        for timestamp in time_index:
            trucks_logging = 0
            for vehicle_id, (first_log, last_log) in vehicle_logging_periods.items():
                if first_log <= timestamp <= last_log:
                    trucks_logging += 1
            df_result.loc[timestamp, 'trucks_logging'] = trucks_logging
        
        df_result = df_result.tz_convert('Europe/Berlin')
        df_result.index.name = 'timestamp'
        # Store the result dataframe for this freight forwarder
        result_dataframes[ff] = df_result

        # Verify if the sum of all occupations equals the number of trucks logging data
        # mismatch = df_result[df_result.loc[:, df_result.columns != 'trucks_logging'].sum(axis=1) != df_result['trucks_logging']]
        # mismatch = mismatch[mismatch]
        # if mismatch.any():
        #     mismatch_minutes = df_result.index[mismatch].tolist()
        #     print(f"Warning: Mismatch found in occupation counts for Freight Forwarder {ff}")
        #     print(f"Mismatch occurred at the following minutes: {mismatch_minutes}")
        # #df_result.loc[:, df_result.columns != 'trucks_logging'].sum(axis=1) == df_result['trucks_logging']
        
        # Print some information about the result
        print(f"Freight Forwarder {ff}:")
        print(f"  Number of vehicles: {num_vehicles}")
        print(f"  Time range: {min_time} to {max_time}")
        print(f"  Shape of result dataframe: {df_result.shape}")
        
        # Calculate and print the maximum number of trucks in each occupation
        max_occupation = df_result[list(occupations)].max()  # Exclude trucks_logging from this calculation
        print("  Maximum trucks by occupation:")
        for occ, max_count in max_occupation.items():
            print(f"    {occ}: {max_count}")
        
        # Print information about trucks logging
        max_logging = df_result['trucks_logging'].max()
        avg_logging = df_result['trucks_logging'].mean()
        print(f"  Maximum trucks logging data simultaneously: {max_logging}")
        print(f"  Average trucks logging data: {avg_logging:.2f}")
        print()
    
    # Save each dataframe to a CSV file
    for ff, df in result_dataframes.items():
        df.to_csv(f'output/csvs/minute_occupation_{ff}.csv')
    
    return result_dataframes


# ------------------------------------------------------------------------------
#                  SCENARIO 2 - No DISPOSITION - INPUT GENERATION
# ------------------------------------------------------------------------------


def tracks_energy_con_and_regen(df_activities, charging_powers, batt_cap, soc_min, soc_max, **kwargs):
    """
    Adds battery energy, state of charge (SoC), and energy recharged columns to the activities dataframe. 
    Unlike sq.truck_soc() this function does not consider truck disposition and only looks at each tour separately.
    
    Parameters:
    -----------
    df_activities : pandas.DataFrame
        DataFrame containing activities (driving and rest periods)
    charging_powers : dict
        Dictionary with occupation types as keys and charging powers (kW) as values
    soc_min : float, optional
        Minimum acceptable state of charge for reporting statistics
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added battery_energy_kwh, energy_recharged_kwh, energy_recharged_kwh_potential, and soc columns
    """
    # Create a copy to avoid modifying the original
    df = df_activities.copy()
    
    # Sort by tour_id and start_time to ensure proper sequence
    df = df.sort_values(['tour_id', 'start_time'])
    
    # Add battery_energy, soc, and energy_recharged columns
    df['energy_recharged_kwh'] = None               # Considering the maximum battery capacity of the truck
    df['energy_recharged_kwh_potential'] = None     # Maximum recharge potential during the rest period given bigger battery size
    df['battery_energy_kwh'] = None
    df['soc'] = None
    
    # Define battery capacity
    max_battery_energy = batt_cap * soc_max
    
    # Process each tour separately
    for tour_id in df['tour_id'].unique():
        # Get all activities for this tour
        tour = df[df['tour_id'] == tour_id].copy()
        
        # Initialize battery energy for first activity
        current_battery_energy = max_battery_energy  # Start at 90% SoC
        
        # Process each activity in sequence
        for idx in tour.index:
            activity = tour.loc[idx]
            
            if activity['occupation'] == 'driving':
                # Subtract energy consumption for driving
                if 'energy_consumption_kwh_cleaned' in df.columns and pd.notna(activity['energy_consumption_kwh_cleaned']):
                    current_battery_energy -= activity['energy_consumption_kwh_cleaned']
                    current_battery_energy = min(current_battery_energy, max_battery_energy)
                
                # Set the starting battery energy for this drive
                df.at[idx, 'battery_energy_kwh'] = current_battery_energy
                
                # No energy recharged during driving (keep as None)
                # df.at[idx, 'energy_recharged_kwh'] remains None

            else:
                # For rest periods, check if charging is available at this occupation
                occupation = activity['occupation']
                if occupation in charging_powers and pd.notna(activity['duration_h']):
                    charging_power = charging_powers[occupation]  # kW
                    charging_time = activity['duration_h']  # hours
                    energy_charged = charging_power * charging_time  # kWh
                    
                    # Calculate how much energy is actually added (limited by battery capacity)
                    energy_added_pot = min(energy_charged, max_battery_energy - current_battery_energy)
                    energy_added = min(energy_charged, max_battery_energy - max(current_battery_energy, batt_cap*soc_min))
                    
                    # Add charged energy, but don't exceed max battery energy
                    current_battery_energy = min(current_battery_energy + energy_charged, max_battery_energy)

                    # Record recharged energy only for industrial area
                    if occupation == 'industrial area' and energy_added > 0:
                        df.at[idx, 'energy_recharged_kwh'] = energy_added
                        df.at[idx, 'energy_recharged_kwh_potential'] = energy_added_pot
                    # Otherwise leave as None

                df.at[idx, 'battery_energy_kwh'] = current_battery_energy
            
            # Calculate SoC
            df.at[idx, 'soc'] = current_battery_energy / batt_cap
    
    for tour_id in df['tour_id'].unique():
        # Get all activities for this tour
        tour = df[df['tour_id'] == tour_id].copy()
        tour['soc_start'] = tour['soc'].shift(1)
        tour.loc[tour.index[0], 'soc_start'] = soc_max  # Set soc_max for the first activity
        # Update the main dataframe
        df.loc[tour.index, 'soc_start'] = tour['soc_start']      


    # Round to reasonable precision
    df['battery_energy_kwh'] = pd.to_numeric(df['battery_energy_kwh'])
    df['energy_recharged_kwh'] = pd.to_numeric(df['energy_recharged_kwh'])
    df['energy_recharged_kwh_potential'] = pd.to_numeric(df['energy_recharged_kwh_potential'])
    df['soc'] = pd.to_numeric(df['soc'])
    df['battery_energy_kwh'] = df['battery_energy_kwh'].round(2)
    
    # Only round energy_recharged_kwh where it's not None
    mask = df['energy_recharged_kwh'].notna()
    if mask.any():
        df.loc[mask, 'energy_recharged_kwh'] = df.loc[mask, 'energy_recharged_kwh'].round(2)
        df.loc[mask, 'energy_recharged_kwh_potential'] = df.loc[mask, 'energy_recharged_kwh_potential'].round(2)

    df['soc'] = df['soc'].round(4)
    df['soc_start'] = df['soc_start'].astype(float).round(4)

    # Calculate the number of instances where soc < soc_min for each freight forwarder
    negative_soc_counts = df[df['soc'] < soc_min].groupby('freight_forwarder').size()
    total_counts = df.groupby('freight_forwarder').size()
    negative_soc_percentage = (negative_soc_counts / total_counts) * 100

    for ff in negative_soc_counts.index:
        print(f"Freight Forwarder {ff}:")
        print(f"  Instances with soc < soc_min: {negative_soc_counts[ff]}")
        print(f"  Percentage: {negative_soc_percentage[ff]:.2f}% \n")
    
    # Print total energy recharged at industrial areas
    total_recharged = df['energy_recharged_kwh'].sum()
    print(f"Total energy recharged at industrial areas: {total_recharged:.2f} kWh")
    # Print total energy recharged potential at industrial areas
    total_recharged_potential = df['energy_recharged_kwh_potential'].sum()
    print(f"Total energy recharged potential at industrial areas: {total_recharged_potential:.2f} kWh")

    # Save the updated dataframe to a CSV file
    # df.to_csv(f"output/track_energies/activities_constant_charging_{charging_powers['home base']}-{charging_powers['industrial area']}_no_disp.csv", index=False)
    
    return df


# ------------------------------------------------------------------------------
#                              DATA QUALITY
# ------------------------------------------------------------------------------


def preprocess_data_quality(df_trips, df_fleet):
    df_track_gap = df_trips.groupby('vehicle_id').sum() 
    df_track_gap['track_gap_km'] = df_track_gap.track_gap / 1000
    df_track_gap['ratio'] = df_track_gap.track_gap / df_track_gap.distance

    return df_track_gap
