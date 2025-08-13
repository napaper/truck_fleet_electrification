"""
Energy simulation functions for truck fleet electrification analysis.

This module provides functions for running energy consumption simulations on GPS track data,
including data quality checks, energy simulation execution, and data cleaning operations.

The module integrates with the BETOS framework for energy consumption modeling and handles
track data validation to ensure simulation quality.
"""

import os
import pandas as pd
import numpy as np
import sqlalchemy
from datetime import timedelta

from energy_sim.BETOS_Framework.scenario_definition import scenario_gen
from energy_sim.Simulation_Energy_Consumption import sim_energy_con


# =============================================================================
# DATA QUALITY VALIDATION FUNCTIONS
# =============================================================================

def faulty_tracks():
    """
    Identify and analyze tracks with data quality issues.
    
    This function checks for tracks with significant data gaps, mismatched timestamps,
    or other quality issues that could affect energy simulation accuracy.
    
    Returns:
        None: Saves results to '../output/mismatched_tracks.csv'
        
    Note:
        This function requires database access and is part of the legacy workflow.
        Consider updating to use CSV-based data sources for consistency.
    """
    print("Checking for faulty tracks")
    
    # Load track data and mapping information
    tracks_filtered = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input/stations/tracks_filtered.csv'))
    track_id_map = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input/stations/track_ids.csv'))

    # Filter tracks longer than 5 km for analysis
    track_ids = tracks_filtered[tracks_filtered['distance_km'] > 5]['track_id'].values
    original_track_ids = np.array([track_id_map.loc[track_id] for track_id in track_ids])
    
    # Create DataFrame with track timing information
    track_times = pd.DataFrame({
        'track_id': tracks_filtered['track_id'],
        'start_time': pd.to_datetime(tracks_filtered['start_time'], format='mixed'),
        'stop_time': pd.to_datetime(tracks_filtered['stop_time'], format='mixed')
    })
    
    # Database connection for GPS data retrieval
    engine = sqlalchemy.create_engine('postgresql://{username}:{password}@{url}:{port}/{db_name}'.format(
            username='ftm_asc_zaehringer',
            url='postgres-sm.ftm.ed.tum.de',
            password='JurmMc2Jjj4uCJpckhf',
            db_name='mobtrack',
            port=5432))

    mismatched_tracks = []

    # Analyze each track for data quality issues
    for track_id_new, track_id in original_track_ids:
        # Retrieve GPS data for the track
        track = pd.read_sql(
            f'''select _time as time, altitude, speed from public.ftm_reduced_gps_from_track_id({str(track_id)},1)''', 
            con=engine, 
            index_col='time'
        ) 
        
        # Extract timing information for the current track
        start_time = track_times.loc[track_times['track_id'] == track_id_new, 'start_time'].values[0]
        stop_time = track_times.loc[track_times['track_id'] == track_id_new, 'stop_time'].values[0]
        
        # Normalize timestamps: round to nearest second and remove timezone info
        # GPS data (1 Hz) is missing timezone information
        start_time = start_time.replace(microsecond=0).replace(tzinfo=None)
        stop_time = stop_time.replace(microsecond=0).replace(tzinfo=None)

        # Check for data gaps in the time series
        time_diffs = track.index.to_series().diff().dropna()
        is_constant_interval = (time_diffs == pd.Timedelta(seconds=1)).all()

        total_duration = (stop_time - start_time).total_seconds()
        
        # Analyze data gaps if intervals are not constant
        if not is_constant_interval:
            # Remove 1s from each timedelta to account for correct 1s differences
            # Only consider abnormal time differences between datapoints
            adjusted_diffs = time_diffs - pd.Timedelta(seconds=1)
            largest_gap = adjusted_diffs.max()
            gap_ratio = adjusted_diffs.sum().total_seconds() / total_duration

            # Determine if gaps are significant enough to skip simulation
            if gap_ratio > 0.02 or largest_gap > timedelta(seconds=30):
                print(f"Energy simulation for Track ID {track_id_new} is skipped, due to a data gap of {largest_gap.total_seconds()} seconds.")
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': gap_ratio, 
                    'total_diff': adjusted_diffs.sum().total_seconds(), 
                    'total_duration': total_duration,
                    'simulation_skipped': True,
                    'gap': True
                })
                continue
            else:
                # Track has gaps but they are acceptable
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': gap_ratio, 
                    'total_diff': adjusted_diffs.sum().total_seconds(), 
                    'total_duration': total_duration,
                    'simulation_skipped': False,
                    'gap': True
                })

        # Check for timestamp mismatches at start/end of tracks
        start_diff = abs((track.index[0] - start_time).total_seconds())
        stop_diff = abs((track.index[-1] - stop_time).total_seconds())
        timestamp_ratio = (start_diff + stop_diff) / total_duration

        # Allow 1 second tolerance for timestamp alignment
        has_timestamp_mismatch = (
            (track.index[0] != start_time and track.index[0] != start_time + timedelta(seconds=1)) or
            (track.index[-1] != stop_time and track.index[-1] != stop_time + timedelta(seconds=1))
        )
        
        if has_timestamp_mismatch:
            print(f"Track ID {track_id_new} has mismatched timestamps")
            print(f"Track ID {track_id_new} has a ratio of {timestamp_ratio:.2f} between the sum of timedeltas and the total duration of the track.")

            # Determine if mismatches are significant enough to skip simulation
            if timestamp_ratio > 0.02 or (start_diff + stop_diff) > 180 or (timestamp_ratio + gap_ratio) > 0.02:
                print(f"Energy simulation for Track ID {track_id_new} is skipped")
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': timestamp_ratio, 
                    'total_diff': start_diff + stop_diff, 
                    'total_duration': total_duration,
                    'simulation_skipped': True, 
                    'gap': False 
                })
                continue
            else:
                # Track has timestamp mismatches but they are acceptable
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': timestamp_ratio, 
                    'total_diff': start_diff + stop_diff, 
                    'total_duration': total_duration,
                    'simulation_skipped': False, 
                    'gap': False
                })

    # Save analysis results to CSV file
    mismatched_tracks_df = pd.DataFrame(mismatched_tracks)
    mismatched_tracks_df.to_csv(os.path.join(os.path.dirname(__file__), '../output/mismatched_tracks.csv'))


# =============================================================================
# ENERGY SIMULATION EXECUTION
# =============================================================================

def run_energy_sim(override_mismatched_tracks=False):
    """
    Execute energy consumption simulation for all valid tracks.
    
    This function runs the energy simulation using the BETOS framework for each track
    that passes data quality checks. It processes tracks sequentially and updates
    the main dataset with energy consumption results.
    
    Args:
        override_mismatched_tracks (bool): Whether to override and save updated
            mismatched tracks data. Default is False.
            
    Returns:
        pd.DataFrame: Updated tracks data with energy consumption information
        
    Note:
        This function can be modified to run simulations on specific track subsets
        by uncommenting and modifying the track_ids array.
    """
    print('Simulating energy consumption')
    
    # Load required data files
    tracks_filtered = pd.read_csv('input/stations/tracks_filtered.csv')
    mismatched_tracks_df = pd.read_csv('output/csvs/mismatched_tracks.csv')

    # Merge duplicate track entries and consolidate quality metrics
    # A tour could be included twice if it has both a datagap and mismatched start/stop times
    duplicate_track_ids = mismatched_tracks_df[mismatched_tracks_df.duplicated(subset=['track_id'], keep=False)]
    
    # Aggregate duplicate rows by track_id
    merged_tracks = duplicate_track_ids.groupby('track_id').agg({
        'ratio': 'sum',
        'total_diff': 'sum',
        'total_duration': 'mean',
        'simulation_skipped': 'max',  # max returns True if any row has True
        'gap': 'max'
    }).reset_index()

    # Replace duplicate rows with consolidated information
    for track_id in merged_tracks['track_id']:
        mismatched_tracks_df = mismatched_tracks_df[mismatched_tracks_df['track_id'] != track_id]
        mismatched_tracks_df = pd.concat([mismatched_tracks_df, merged_tracks[merged_tracks['track_id'] == track_id]], ignore_index=True)

    mismatched_tracks_df = mismatched_tracks_df.sort_values(by='track_id').reset_index(drop=True)

    # Save updated mismatched tracks data if requested
    if override_mismatched_tracks:
        mismatched_tracks_df.to_csv(os.path.join(os.path.dirname(__file__), '../output/mismatched_tracks.csv'), mode='w')
    
    # Identify tracks to skip based on quality criteria
    tracks_to_skip = mismatched_tracks_df[mismatched_tracks_df['simulation_skipped'] == True]
    tracks_to_skip = tracks_to_skip['track_id'].values
    
    # Filter tracks: remove those shorter than 5 km or with significant data gaps
    track_ids = tracks_filtered[tracks_filtered['distance_km'] > 5]['track_id'].values
    track_ids = np.setdiff1d(track_ids, tracks_to_skip)

    # Configuration for subset simulation (useful for testing or specific analysis)
    # To run simulation for a subset of tracks, uncomment and modify the following line:
    # track_ids = np.array([10786, 23894, 119905, 119947, 126509, 144124, 144175, 148066])
    # track_ids = np.array([126509])  # Single track example

    # Load track ID mapping for database queries
    track_id_map = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input/stations/track_ids.csv'))
    original_track_ids = np.array([track_id_map.loc[track_id] for track_id in track_ids])

    zero_speed_tracks = []

    # Execute energy simulation for each valid track
    for track_id_new, track_id in original_track_ids:
        print(f"Processing track ID: {track_id_new}")

        # Generate scenario and run energy simulation
        track_scenario = scenario_gen(track_id)
        sim = sim_energy_con(track_scenario)

        # Update tracks data with energy consumption results
        tracks_filtered.loc[
            tracks_filtered['track_id'] == track_id_new, 'energy_consumption_kwh'
        ] = sim.energy_consumption_kwh

        # Handle tracks with no energy consumption (zero speed)
        if sim.energy_consumption_kwh is None:
            zero_speed_tracks.append(track_id_new)
            continue

        # Calculate average energy consumption per kilometer
        tracks_filtered.loc[
            tracks_filtered['track_id'] == track_id_new, 'avg_energy_consumption_kwh/km'
        ] = (sim.energy_consumption_kwh / tracks_filtered.loc[
            tracks_filtered['track_id'] == track_id_new, 'distance_km'
        ]) 
        
        # Intermediate saves for long-running simulations (commented out by default)
        # Uncomment these lines if you need to save progress during long simulations
        # if track_id_new == 50988:
        #     tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_filtered_with_energy1.csv'))
        # if track_id_new == 102899:
        #     tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_filtered_with_energy2.csv'))
        # if track_id_new == 129210:
        #     tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_filtered_with_energy3.csv'))

    # Save final results and zero-speed track information
    # tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_with_energy_raw.csv'))
    zero_speed_tracks_df = pd.DataFrame(zero_speed_tracks, columns=['track_id'])
    zero_speed_tracks_df.to_csv(os.path.join(os.path.dirname(__file__), '../output/zero_speed_tracks.csv'))

    return tracks_filtered


# =============================================================================
# DATA CLEANING AND POST-PROCESSING
# =============================================================================

def clean_energy_sim_data(trips=None):
    """
    Clean and post-process energy simulation data.
    
    This function rounds energy consumption values to consistent decimal places
    and fills missing values using statistical averages to ensure data completeness.
    
    Args:
        trips (pd.DataFrame, optional): Input trips data. If None, loads from
            'output/csvs/tracks_with_energy_raw.csv'. Default is None.
            
    Returns:
        pd.DataFrame: Cleaned trips data with additional processed columns
        
    Note:
        The function creates two new columns: 'energy_consumption_kwh_cleaned' and
        'avg_energy_consumption_kwh/km_cleaned' with filled missing values.
    """
    # Load data if not provided
    if trips is None:
        trips = pd.read_csv('output/csvs/tracks_with_energy_raw.csv', index_col='track_id')

    # Round energy consumption values to 6 decimal places for consistency
    trips['energy_consumption_kwh'] = trips['energy_consumption_kwh'].round(6)
    trips['avg_energy_consumption_kwh/km'] = trips['avg_energy_consumption_kwh/km'].round(6)

    # Calculate statistical averages for missing value imputation
    avg_total_consumption = trips['energy_consumption_kwh'].mean().round(6)
    avg_consumption_per_km = trips['avg_energy_consumption_kwh/km'].mean().round(6)

    # Create cleaned columns with imputed missing values
    # Use average consumption per km to estimate total energy where missing
    trips['energy_consumption_kwh_cleaned'] = trips.apply(
        lambda row: avg_consumption_per_km * row['distance_km'] 
        if pd.isna(row['energy_consumption_kwh']) 
        else row['energy_consumption_kwh'], 
        axis=1
    )
    
    # Fill missing average consumption values with the overall average
    trips['avg_energy_consumption_kwh/km_cleaned'] = trips['avg_energy_consumption_kwh/km'].fillna(avg_consumption_per_km)
    
    # Round cleaned values to 6 decimal places for consistency
    trips['energy_consumption_kwh_cleaned'] = trips['energy_consumption_kwh_cleaned'].round(6)
    trips['avg_energy_consumption_kwh/km_cleaned'] = trips['avg_energy_consumption_kwh/km_cleaned'].round(6)

    # Save cleaned data to CSV file
    trips.to_csv(os.path.join(os.path.dirname(__file__), '../output/csvs/tracks_with_energy.csv'))
    
    return trips


