import os
import pandas as pd
import numpy as np
import sqlalchemy
from datetime import timedelta

from energy_sim.BETOS_Framework.scenario_definition import scenario_gen
from energy_sim.Simulation_Energy_Consumption import sim_energy_con


def faulty_tracks():
    print("Checking for faulty tracks")
    tracks_filtered = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input/stations/tracks_filtered.csv'))
    track_id_map = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input/stations/track_ids.csv'))

    # remove tracks, < 5 km 
    track_ids = tracks_filtered[tracks_filtered['distance_km'] > 5]['track_id'].values
    original_track_ids = np.array([track_id_map.loc[track_id] for track_id in track_ids])
    track_times = pd.DataFrame({
        'track_id': tracks_filtered['track_id'],
        'start_time': pd.to_datetime(tracks_filtered['start_time'], format='mixed'),   #.dt.round('S'),
        'stop_time': pd.to_datetime(tracks_filtered['stop_time'], format='mixed')      #.dt.round('S')
        })
    
#TODO: change to use config file / gitignore
    
    engine = sqlalchemy.create_engine('postgresql://{username}:{password}@{url}:{port}/{db_name}'.format(
            username='ftm_asc_paper',   #'ftm_asc_zaehringer',
            url='postgres-sm.ftm.ed.tum.de', #postgres-sm.ftm.ed.tum.de
            password='u436UDyanxDuDiMUpATu5eKb',   #'JurmMc2Jjj4uCJpckhf',
            db_name='mobtrack',
            port=5432))

    mismatched_tracks = []


    for track_id_new, track_id in original_track_ids:
        track = pd.read_sql(
        f'''select _time as time, altitude, speed from public.ftm_reduced_gps_from_track_id({str(track_id)},1)''', 
        con=engine, 
        index_col='time'
        ) 
        
        start_time = track_times.loc[track_times['track_id'] == track_id_new, 'start_time'].values[0]
        stop_time = track_times.loc[track_times['track_id'] == track_id_new, 'stop_time'].values[0]
        
        # round to nearest second and remove timezone info, because track (1 Hz Data) is missing timezone info
        start_time = start_time.replace(microsecond=0).replace(tzinfo=None)
        stop_time = stop_time.replace(microsecond=0).replace(tzinfo=None)

        # Calculate the difference between consecutive elements
        diff = track.index.to_series().diff().dropna()

        # Check if all differences are equal to 1 second
        is_constant_interval = (diff == pd.Timedelta(seconds=1)).all()

        total_duration = (stop_time - start_time).total_seconds()
        if not is_constant_interval:
            # remove 1s from each timedelta to account for the (correct) 1s difference between consecutive elements
            # -> only consider abnormal time differences between datapoints 
            diff = diff - pd.Timedelta(seconds=1)
            largest_diff = diff.max()
            gap_ratio = diff.sum().total_seconds() / total_duration

            if gap_ratio > 0.02 or largest_diff > timedelta(seconds=30):
                print(f"Energy simulation for Track ID {track_id_new} is skipped, due to a data gap of {largest_diff.total_seconds()} seconds.")
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': gap_ratio, 
                    'total_diff': diff.sum().total_seconds(), 
                    'total_duration': total_duration,
                    'simulation_skipped': True,
                    'gap': True
                })
                continue

            else:
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': gap_ratio, 
                    'total_diff': diff.sum().total_seconds(), 
                    'total_duration': total_duration,
                    'simulation_skipped': False,
                    'gap': True
                })

        start_diff = abs((track.index[0] - start_time).total_seconds())
        stop_diff = abs((track.index[-1] - stop_time).total_seconds())
        ratio = (start_diff + stop_diff) / total_duration

        if (track.index[0] != start_time and track.index[0] != start_time + timedelta(seconds=1)) or \
        (track.index[-1] != stop_time and track.index[-1] != stop_time + timedelta(seconds=1)):
            print(f"Track ID {track_id_new} has mismatched timestamps")
            print(f"Track ID {track_id_new} has a ratio of {ratio:.2f} between the sum of timedeltas and the total duration of the track.")

            if ratio > 0.02 or (start_diff + stop_diff) > 180 or (ratio + gap_ratio) > 0.02:
                print(f"Energy simulation for Track ID {track_id_new} is skipped")
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': ratio, 
                    'total_diff': start_diff + stop_diff, 
                    'total_duration': total_duration,
                    'simulation_skipped': True, 
                    'gap': False 
                })
                continue

            else:
                mismatched_tracks.append({
                    'track_id': track_id_new, 
                    'ratio': ratio, 
                    'total_diff': start_diff + stop_diff, 
                    'total_duration': total_duration,
                    'simulation_skipped': False, 
                    'gap': False
                })

    mismatched_tracks_df = pd.DataFrame(mismatched_tracks)
    mismatched_tracks_df.to_csv(os.path.join(os.path.dirname(__file__), '../output/mismatched_tracks.csv'))


def run_energy_sim(override_mismatched_tracks=False):
    print('Simulating energy consumption')
    tracks_filtered = pd.read_csv('input/stations/tracks_filtered.csv')

    mismatched_tracks_df = pd.read_csv('output/csvs/mismatched_tracks.csv')

    # first need to merge the mismatched track csvs with each other and remove duplicate tours 
    # a tour could be included twice if it has both a datagap and mismatched start/stop times

    # Extract duplicate track_ids
    duplicate_track_ids = mismatched_tracks_df[mismatched_tracks_df.duplicated(subset=['track_id'], keep=False)]
    # Merge rows with the same track_id
    merged_tracks = duplicate_track_ids.groupby('track_id').agg({
        'ratio': 'sum',
        'total_diff': 'sum',
        'total_duration': 'mean',
        'simulation_skipped': 'max',  # max will return True if any of the rows have True
        'gap': 'max'
    }).reset_index()

    # Replace duplicate rows in mismatched_tracks_df with corresponding rows from merged_tracks
    for track_id in merged_tracks['track_id']:
        mismatched_tracks_df = mismatched_tracks_df[mismatched_tracks_df['track_id'] != track_id]
        mismatched_tracks_df = pd.concat([mismatched_tracks_df, merged_tracks[merged_tracks['track_id'] == track_id]], ignore_index=True)

    mismatched_tracks_df = mismatched_tracks_df.sort_values(by='track_id').reset_index(drop=True)

    # not actually tested if this works
    if override_mismatched_tracks:
        mismatched_tracks_df.to_csv(os.path.join(os.path.dirname(__file__), '../output/mismatched_tracks.csv'), mode='w')
    
    tracks_to_skip = mismatched_tracks_df[mismatched_tracks_df['simulation_skipped'] == True]

    # Remove tracks that are shorter than 5 km or have significant datagaps
    tracks_to_skip = tracks_to_skip['track_id'].values
    track_ids = tracks_filtered[tracks_filtered['distance_km'] > 5]['track_id'].values
    track_ids = np.setdiff1d(track_ids, tracks_to_skip)

    """
    To only run the simulation for a subset of tracks, simply initialize track_ids with the desired track_ids
    This is for example useful when wantiung to plot the altitude and speed profiles of a specific set of tracks
    For example:
    track_ids = np.array([10786, 23894, 119905, 119947, 126509, 144124, 144175, 148066])
    """
    #track_ids = np.array([126509])

    track_id_map = pd.read_csv(os.path.join(os.path.dirname(__file__), '../input/stations/track_ids.csv'))
    original_track_ids = np.array([track_id_map.loc[track_id] for track_id in track_ids])

    zero_speed_tracks = []

    for track_id_new, track_id in original_track_ids:
        print(track_id_new)

        track_scenario = scenario_gen(track_id)
        sim = sim_energy_con(track_scenario)

        tracks_filtered.loc[
            tracks_filtered['track_id'] == track_id_new, 'energy_consumption_kwh'
        ] = sim.energy_consumption_kwh

        if sim.energy_consumption_kwh is None:
            zero_speed_tracks.append(track_id_new)
            continue

        tracks_filtered.loc[
            tracks_filtered['track_id'] == track_id_new, 'avg_energy_consumption_kwh/km'
        ] = (sim.energy_consumption_kwh / tracks_filtered.loc[
            tracks_filtered['track_id'] == track_id_new, 'distance_km'
        ]) 
        
        # intermediate saves due to long runtime 
        #if track_id_new == 50988:
        #    tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_filtered_with_energy1.csv'))
        #if track_id_new == 102899:
        #    tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_filtered_with_energy2.csv'))
        #if track_id_new == 129210:
        #    tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_filtered_with_energy3.csv'))

    # save tracks_filtered with energy consumption
    #tracks_filtered.to_csv(os.path.join(os.path.dirname(__file__), '../output/tracks_with_energy_raw.csv'))
    zero_speed_tracks_df = pd.DataFrame(zero_speed_tracks, columns=['track_id'])
    zero_speed_tracks_df.to_csv(os.path.join(os.path.dirname(__file__), '../output/zero_speed_tracks.csv'))

    return tracks_filtered



def clean_energy_sim_data(trips=None):  
    if trips is None:
        trips = pd.read_csv('output/csvs/tracks_with_energy_raw.csv', index_col='track_id')

    # Round energy_consumption_kwh and avg_power_kwh/km to 6 decimal places
    trips['energy_consumption_kwh'] = trips['energy_consumption_kwh'].round(6)
    trips['avg_energy_consumption_kwh/km'] = trips['avg_energy_consumption_kwh/km'].round(6)

    avg_total_consumption = trips['energy_consumption_kwh'].mean().round(6)
    avg_consumption = trips['avg_energy_consumption_kwh/km'].mean().round(6)

    # Create cleaned columns with original values where available, 
    # otherwise use the average of all simulated energies to calculate the avg energy consumption per km
    # and the newly calculated average energy consumption per km to calculate the total energy consumption
    trips['energy_consumption_kwh_cleaned'] = trips.apply(
        lambda row: avg_consumption * row['distance_km'] 
        if pd.isna(row['energy_consumption_kwh']) 
        else row['energy_consumption_kwh'], 
        axis=1
    )
    trips['avg_energy_consumption_kwh/km_cleaned'] = trips['avg_energy_consumption_kwh/km'].fillna(avg_consumption)
    
    trips['energy_consumption_kwh_cleaned'] = trips['energy_consumption_kwh_cleaned'].round(6)
    trips['avg_energy_consumption_kwh/km_cleaned'] = trips['avg_energy_consumption_kwh/km_cleaned'].round(6)

    trips.to_csv(os.path.join(os.path.dirname(__file__), '../output/csvs/tracks_with_energy.csv'))
    
    return trips


