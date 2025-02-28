import pandas as pd
import numpy as np
import zipfile

LOCATIONS = ['home_base', 'rest_area', 'service_area_fuel', 'industrial_area']

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

color_list = list(colors.values())
colors_fleets = {i: color_list[i - 1] for i in range(1, 5)}
colors_locations = {loc: color_list[i + 4] for i, loc in enumerate([*LOCATIONS, 'other_area', 'driving'])}
colors_locations_2 = {loc.replace('_', ' '): color_list[i + 4] for i, loc in enumerate([*LOCATIONS, 'other_area', 'driving'])}

alpha_major = 0.8
alpha_minor = 0.5

def load_data():
    df_trips_unfiltered = pd.read_csv('input/stations/tracks.csv', index_col='track_id', parse_dates=['start_time', 'stop_time'])
    df_fleet = pd.read_csv('input/home/fleet.csv', index_col='vehicle_id')
    zf = zipfile.ZipFile('input/home/speed.zip')
    return df_trips_unfiltered, df_fleet, zf

def load_speed_data(zf, i_veh, i_trip):
    df_speed = pd.read_csv(zf.open(f'input/home/speed/{i_veh}/{i_trip}.csv'))
    return df_speed

def preprocess_trips_data(df_trips_unfiltered, df_fleet):
    """
    Filters out trips shorter than 1 km or with avg. speeds > max. speed, converts distances to kilometers and speed to km/h, 
    converts time strings to datetime objects and calculates trip durations in hours
    """
    df_trips = df_trips_unfiltered.copy()
    df_trips = df_trips.loc[df_trips_unfiltered.distance > 1000]
    df_trips = df_trips.loc[df_trips_unfiltered.avg_speed <= df_trips_unfiltered.max_speed]
    df_trips['distance_km'] = df_trips['distance'] / 1000
    #transfer strings to datetime objects
    df_trips['stop_time'] = pd.to_datetime(df_trips['stop_time'], format='mixed')
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], format='mixed')
    df_trips['duration'] = (df_trips['stop_time'] - df_trips['start_time']) / pd.Timedelta(seconds=1)
    df_trips['duration_h'] = df_trips['duration'] / 3600
    df_trips['max_speed_kmh'] = df_trips['max_speed'] * 3.6
    df_trips['avg_speed_kmh'] = df_trips['avg_speed'] * 3.6
    df_trips = df_trips.merge(df_fleet[['gross_vehicle_weight', 'total_mass_with_trailer', 'axle_class']], left_on='vehicle_id', right_index=True, how='left')
    return df_trips

def process_stops_data(df_trips):
    df_stops = df_trips.copy()
    df_stops['service_area_fuel'] = (~df_trips['home_base']) & df_trips['service_area_fuel']
    df_stops['rest_area'] = (~df_trips['home_base']) & (~df_trips['service_area_fuel']) & df_trips['rest_area']
    df_stops['industrial_area'] = (~df_trips['home_base']) & (~df_trips['service_area_fuel']) & (~df_stops['rest_area']) & df_trips['industrial_area']
    df_stops['other_area'] = (~df_trips['home_base']) & (~df_trips['service_area_fuel']) & (~df_stops['rest_area']) & (~df_stops['industrial_area'])
    df_stops['location'] = df_stops[LOCATIONS].idxmax(axis=1).str.replace('_', ' ')
    rest_time = df_trips.groupby('vehicle_id').start_time.shift(-1) - df_trips['stop_time']
    rest_time = rest_time.fillna(pd.Timedelta(seconds=0))
    df_stops = df_stops.assign(rest_time=rest_time / pd.Timedelta(seconds=1))
    # If there is no more rest time (i.e. after the last recorded trip), the rest time is set to 0
    #df_stops = df_stops.assign(rest_time=(df_trips.groupby('vehicle_id').start_time.shift(-1) - df_trips.stop_time) / pd.Timedelta(seconds=1))

    df_stops = df_stops.assign(rest_time_h=df_stops['rest_time'] / 3600)
    return df_stops

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

def seconds_to_time(s):
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(s, unit='s')

# ------------------------------------------------------------------------------
#                              GENERAL DESCRIPTION
# ------------------------------------------------------------------------------

def calculate_meta_data(df_trips_unfiltered, df_trips, df_fleet):
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['stop_time'] = pd.to_datetime(df_trips['stop_time'], utc=True)
    
    total_distance = df_trips['distance'].sum()
    total_time = df_trips['duration'].sum()
    trips_longer_1km = len(df_trips)
    tours = len(df_trips[['vehicle_id', 'tour_id']].drop_duplicates()) # HERE I have more tours that calc in the original repo
    fleet_size = len(df_fleet)
    median_vehicle_driving_distance_km = df_trips.groupby('vehicle_id')['distance'].sum().median() / 1000
    
    avg_speed_per_trip = df_trips['avg_speed'].mean()
    max_speed = df_trips['max_speed'].max()
    total_signal_loss = df_trips['n_signal_loss'].sum()
    avg_signal_loss_ratio = df_trips['r_signal_loss'].mean()
    
    avg_hdop = df_trips['avg_hdop'].mean() if 'avg_hdop' in df_trips.columns else None
    max_hdop = df_trips['max_hdop'].max() if 'max_hdop' in df_trips.columns else None
    
    rest_time_stats = df_trips['duration_h'].describe()
    distance_stats = df_trips['distance_km'].describe()
    
    location_distribution = df_trips[LOCATIONS].mean() * 100
    
    max_latitude = 0  # Placeholder for maximum latitude
    min_latitude = 0  # Placeholder for minimum latitude
    max_longitude = 0  # Placeholder for maximum longitude
    min_longitude = 0  # Placeholder for minimum longitude
    
    avg_points_per_trip = df_trips['track_gap'].mean()
    
    avg_trip_time_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['duration'].mean().mean()
    max_trip_time_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['duration'].sum().max()
    
    avg_trip_distance_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['distance'].mean().mean()
    max_trip_distance_per_day = df_trips.groupby(df_trips['start_time'].dt.date)['distance'].sum().max()
    
    avg_trips_per_tour = df_trips.groupby('tour_id').size().mean()
    avg_signal_loss_per_tour = df_trips.groupby('tour_id')['r_signal_loss'].mean().mean()
    
    total_recordings = len(df_trips_unfiltered)
    recordings_under_1km = len(df_trips_unfiltered[df_trips_unfiltered['distance'] < 1000])  
    
    min_start_time = df_trips['start_time'].min()
    max_start_time = df_trips['start_time'].max()
    
    # Temporal meta data
    avg_trips_per_day_per_truck = df_trips.groupby('vehicle_id').size().mean()
    avg_trip_duration_h = df_trips['duration_h'].mean()
    avg_trip_distance_km = df_trips['distance_km'].mean()
    
    # Calculate average daily distance per truck
    df_trips['date'] = df_trips['start_time'].dt.date
    daily_distances = df_trips.groupby(['vehicle_id', 'date'])['distance_km'].sum().reset_index()
    avg_daily_distance_per_truck_km = daily_distances.groupby('vehicle_id')['distance_km'].mean().mean()

    avg_speed_per_trip_kmh = avg_speed_per_trip * 3.6

    # Calculate average start and end time of day
    first_trips = df_trips.groupby(['vehicle_id', 'date'])['start_time'].min().reset_index()
    last_trips = df_trips.groupby(['vehicle_id', 'date'])['stop_time'].max().reset_index()

    avg_daily_start_time_seconds = first_trips['start_time'].dt.hour.mean() * 3600 + \
                                   first_trips['start_time'].dt.minute.mean() * 60 + \
                                   first_trips['start_time'].dt.second.mean()
    avg_daily_end_time_seconds = last_trips['stop_time'].dt.hour.mean() * 3600 + \
                                 last_trips['stop_time'].dt.minute.mean() * 60 + \
                                 last_trips['stop_time'].dt.second.mean()
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
        'avg_speed_per_trip_kmh': avg_speed_per_trip_kmh
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

    general_df = pd.DataFrame(general_data)
    temporal_df = pd.DataFrame(temporal_data)
    spatial_df = pd.DataFrame(spatial_data)
    return general_df, temporal_df, spatial_df



def calculate_weekly_distances(df_trips):
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['start_time'] = df_trips['start_time'].dt.tz_localize(None)
    df_trips['start_time_7d'] = df_trips['start_time'].dt.to_period("W-SUN")
    distances_km = df_trips.groupby(['start_time_7d'])['distance'].sum() / 1000
    return distances_km


 # speed converted to kilometers per hour for visualization
def convert_speed_to_kmh(df_speed):
    df_speed['kmh'] = df_speed['speed'] * 3.6

    # relative time in seconds since beginning of trip
    df_speed['rel_time'] = df_speed.epoch - df_speed.epoch.min()
    df_speed.set_index('rel_time', inplace=True)

    return df_speed


def calculate_occupation(df_trips):
    # TODO Assigns negative duration to trips that are not driving, on purpose?
    # Creates 2x the amount of rows as the original df_trips (as it adds a new occupation between each trip (occupation = driving))

    df_occupation = df_trips[['vehicle_id', 'start_time', 'stop_time']].copy()
    df_occupation.rename(columns={'start_time': 'stop_time', 'stop_time': 'start_time'}, inplace=True)
    
    for loc in [*LOCATIONS, 'other_area']:
        if loc not in df_trips.columns:
            df_trips[loc] = False

    df_occupation = df_occupation.assign(occupation=df_trips[[*LOCATIONS, 'other_area']].idxmax(axis=1))

    df_driving_occupation = df_trips[['vehicle_id', 'start_time', 'stop_time']].copy()
    df_driving_occupation['occupation'] = 'driving'

    df_occupation = pd.concat([df_occupation, df_driving_occupation.reset_index(drop=True)])
    #df_occupation = df_occupation.append(df_driving_occupation.reset_index(drop=True))

    df_occupation['start_time'] = pd.to_datetime(df_occupation['start_time'], utc=True)
    df_occupation['stop_time'] = pd.to_datetime(df_occupation['stop_time'], utc=True)

    df_occupation['duration'] = df_occupation.stop_time - df_occupation.start_time
    df_occupation['duration'] = df_occupation.duration.dt.total_seconds() / 3600
    
    df_occupation.set_index('start_time', inplace=True)
    df_occupation = df_occupation.tz_convert('Europe/Berlin') # This does only converts the timezone of the index (start_time)!!

    return df_occupation


def prepare_occupation_data(df_occupation):
    resampled = df_occupation.groupby(['vehicle_id']).resample('1min').ffill()[['occupation', 'duration']]
    resampled = resampled.reset_index()

    resampled['dow'] = resampled.start_time.dt.dayofweek
    resampled['hour'] = resampled.start_time.dt.hour
    resampled = resampled.loc[resampled.dow < 6] # only week days
    resampled = resampled.loc[resampled.duration < 24] # remove very long stays

    truck_day = resampled.groupby(['occupation', 'hour']).occupation.count()
    truck_day = truck_day.unstack(level=-1, fill_value=0)
    return truck_day


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

    df_rt_joined_plot = df_rt_joined.join(df_driving).rename(columns={'duration': 'driving'})

    return df_rt_joined_plot


def aggregate_driving_times(df_trips, df_rt_joined):
    df_driving = df_trips.groupby('vehicle_id')['duration'].sum()
    df_rt_joined_plot = df_rt_joined.join(df_driving).rename(columns={'duration': 'driving'})
    return df_rt_joined_plot


def calculate_weekly_distances(df_trips):
    df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
    df_trips['start_time'] = df_trips['start_time'].dt.tz_localize(None)
    df_trips['start_time_7d'] = df_trips['start_time'].dt.to_period("W-SUN")

    distances_km = df_trips.groupby(['start_time_7d'])['distance'].sum() / 1000
    return distances_km


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

def aggregate_tours(df_trips):
    df_tours = df_trips.groupby('tour_id').agg({
        'distance': 'sum',
        'distance_km': 'sum',
        'duration': 'sum',
        'duration_h': 'sum',
        'start_time': 'min',
        **{col: 'max' for col in df_trips.columns if col not in ['distance', 'distance_km', 'duration', 'duration_h', 'start_time', 'tour_id']}
    }).reset_index()

    df_tours.to_csv('input/stations/tours.csv', index=False)
    return df_tours


# ------------------------------------------------------------------------------
#                              SPATIAL PATTERN
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
#                              DATA QUALITY
# ------------------------------------------------------------------------------


def preprocess_data_quality(df_trips, df_fleet):
    df_track_gap = df_trips.groupby('vehicle_id').sum() 
    df_track_gap['track_gap_km'] = df_track_gap.track_gap / 1000
    df_track_gap['ratio'] = df_track_gap.track_gap / df_track_gap.distance

    return df_track_gap

