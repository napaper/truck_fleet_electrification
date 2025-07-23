import pandas as pd
import numpy as np
import joblib

cache = joblib.Memory(".cache")  # , verbose=0

@cache.cache
def combine_tracks_and_stops(df_stops, df_tracks_with_energy):
    """
    Processes stops data which contains both track (driving) and stop (rest) information.
    Creates alternating driving and rest activities. Ensures cid is included in all activities.
    The last occupation of each tour_id is set to 'home base'.
    
    Parameters:
    -----------
    df_stops : pandas.DataFrame
        DataFrame containing both track and stop information with 'location' column for stops
    df_tracks_with_energy : pandas.DataFrame, optional
        Not used in this version as df_stops contains all necessary information
    
    Returns:
    --------
    pandas.DataFrame
        Combined dataframe with alternating driving and rest periods
    """
    # Create a copy to avoid modifying the original
    df_stops = df_stops.copy()

    # If df_tracks_with_energy is provided, add missing columns to df_stops
    if df_tracks_with_energy is not None:
        df_stops['energy_consumption_kwh_cleaned'] = df_tracks_with_energy['energy_consumption_kwh_cleaned']
        df_stops['avg_energy_consumption_kwh/km_cleaned'] = df_tracks_with_energy['avg_energy_consumption_kwh/km_cleaned']
    
    # Create a list to store all activities (driving and rests)
    all_activities = []
    
    # Get unique vehicle IDs
    vehicles = df_stops['vehicle_id'].unique()
    
    # Process each vehicle separately
    for vehicle_id in vehicles:
        print(f' Vehicle ID: {vehicle_id}')
        # Get all stops for this vehicle, sorted by start time
        vehicle_data = df_stops[df_stops['vehicle_id'] == vehicle_id].sort_values('start_time')
        
        if len(vehicle_data) == 0:
            continue
        
        # Process each entry
        for i, (idx, entry) in enumerate(vehicle_data.iterrows()):
            # Create a driving activity from the current entry
            driving_activity = entry.copy()
            driving_activity['occupation'] = 'driving'
            driving_activity['track_id'] = idx  # Store the original index as track_id
            driving_activity['cid'] = entry['cid']
            
            # Keep only the specified columns if they exist in the DataFrame
            columns_to_keep = [
                'track_id', 'vehicle_id', 'tour_id', 'start_time', 'stop_time', 'distance',
                'freight_forwarder', 'distance_km', 'duration', 'duration_h', 'max_speed_kmh',
                'avg_speed_kmh', 'location', 'occupation', 'long_haul', 'cid', 
                'energy_consumption_kwh_cleaned', 'avg_energy_consumption_kwh/km_cleaned'
            ]
            driving_activity = driving_activity[driving_activity.index.intersection(columns_to_keep)]
            
            all_activities.append(driving_activity)
            
            # If this entry has location information, create a rest activity
            if 'location' in entry and pd.notna(entry['location']):
                # Use the location as the occupation for the rest period
                occupation = entry['location']
                
                # If there's a next entry, use its start time as the end of the rest period
                if i < len(vehicle_data) - 1 and pd.notna(vehicle_data.iloc[i + 1]['start_time']):
                    rest_end_time = vehicle_data.iloc[i + 1]['start_time']
                    
                    # Create rest activity Series with required fields
                    rest_activity_dict = {
                        'vehicle_id': vehicle_id,
                        'track_id': None,  # No track_id for rest periods
                        'start_time': entry['stop_time'],
                        'stop_time': rest_end_time,
                        'occupation': occupation,
                        'freight_forwarder': entry['freight_forwarder'] if 'freight_forwarder' in entry else None,
                        'duration': entry['rest_time'],
                        'duration_h': entry['rest_time_h'],
                        'location': entry['location'],
                        'tour_id': entry['tour_id'],
                        'long_haul': entry['long_haul']
                    }
                    
                    rest_activity_dict['cid'] = entry['cid']
                    
                    rest_activity = pd.Series(rest_activity_dict)
                    all_activities.append(rest_activity)
                    
                    # Store this as the last activity for this tour_id
                    tour_id = entry['tour_id']
    
    # Convert the list to a DataFrame
    df_activities = pd.DataFrame(all_activities)

    # Ensure tour_id is numeric
    df_activities['tour_id'] = pd.to_numeric(df_activities['tour_id'], downcast='integer')
    
    # Sort by vehicle_id and start_time
    df_activities = df_activities.sort_values(['vehicle_id', 'start_time'])
    
    # Remove any duplicate columns
    df_activities = df_activities.loc[:, ~df_activities.columns.duplicated()]
    
    # Override the occupation of the last activity of each tour_id with 'home base'
    # First, get the last activity for each tour_id
    last_activities = df_activities.groupby('tour_id').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    
    # Set the occupation to 'home base' for these activities
    for idx in last_activities.index:
        tour_id = last_activities.loc[idx, 'tour_id']
        # Find all rows with this tour_id
        tour_activities = df_activities[df_activities['tour_id'] == tour_id]
        # Get the index of the last activity
        last_idx = tour_activities[tour_activities.stop_time == tour_activities.stop_time.max()].index
        # Set the occupation to 'home base'
        df_activities.loc[last_idx, 'occupation'] = 'home base'

    # Save the combined dataframe to a CSV file
    # df_activities.to_csv('output/csvs/tracks_and_stops.csv', index=False)
    
    # Reset index
    df_activities = df_activities.reset_index(drop=True)
    
    return df_activities


@cache.cache
def truck_soc(df_activities, charging_powers, batt_cap, soc_min, soc_max, **kwargs):
    import numpy as np
    import pandas as pd

    df = df_activities.copy()
    df = df.sort_values(['vehicle_id', 'start_time'])

    df['energy_recharged_kwh'] = None     
    df['battery_energy_kwh'] = None
    df['soc'] = None

    max_battery_energy = batt_cap * soc_max
    min_battery_energy = batt_cap * soc_min

    for vehicle_id in df['vehicle_id'].unique():
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        current_battery_energy = max_battery_energy

        for idx in vehicle_data.index:
            activity = vehicle_data.loc[idx]
            energy_added = 0

            if activity['occupation'] == 'driving':
                if 'energy_consumption_kwh_cleaned' in df.columns and pd.notna(activity['energy_consumption_kwh_cleaned']):
                    current_battery_energy -= activity['energy_consumption_kwh_cleaned']
                    # Emergency top-up to prevent below min SoC
                    if current_battery_energy < min_battery_energy:
                        energy_needed = min_battery_energy - current_battery_energy
                        current_battery_energy = min_battery_energy
                        energy_added = energy_needed  # For tracking

            else:
                occupation = activity['occupation']
                if occupation in charging_powers:
                    charging_power = charging_powers[occupation]
                    charging_time = activity['duration_h']
                    energy_possible_now = charging_power * charging_time
                    energy_to_full = max_battery_energy - current_battery_energy

                    # If this is NOT a home base and NOT driving
                    if occupation != 'home base' and occupation != 'driving':
                        # Check whether we can safely skip charging and reach the next home base with enough SoC
                        future_activities = vehicle_data.loc[idx+1:]
                        cumulative_consumption = 0
                        home_found = False

                        for future_idx, future_activity in future_activities.iterrows():
                            if future_activity['occupation'] == 'driving' and pd.notna(future_activity.get('energy_consumption_kwh_cleaned', None)):
                                cumulative_consumption += future_activity['energy_consumption_kwh_cleaned']
                            elif future_activity['occupation'] == 'home base':
                                home_found = True
                                break

                        if home_found:
                            projected_energy = current_battery_energy - cumulative_consumption
                            if projected_energy >= min_battery_energy:
                                energy_added = 0  # Safe to defer charging
                            else:
                                # Not safe – must charge now to avoid SoC dip
                                required_now = min_battery_energy + cumulative_consumption - current_battery_energy
                                energy_added = min(energy_possible_now, required_now, energy_to_full)
                                current_battery_energy += energy_added
                        else:
                            # No home charging ahead – charge as much as possible now
                            energy_added = min(energy_possible_now, energy_to_full)
                            current_battery_energy += energy_added
                    else:
                        # Home base or driving: charge as usual
                        energy_added = min(energy_possible_now, energy_to_full)
                        current_battery_energy += energy_added

                    # Enforce bounds
                    assert current_battery_energy >= min_battery_energy - 1e-3, (
                        f"Energy dropped below min: {current_battery_energy:.2f} < {min_battery_energy:.2f}"
                    )
                    assert current_battery_energy <= max_battery_energy + 1e-3, (
                        f"Energy exceeded max: {current_battery_energy:.2f} > {max_battery_energy:.2f}"
                    )

            df.at[idx, 'battery_energy_kwh'] = current_battery_energy
            df.at[idx, 'energy_recharged_kwh'] = energy_added
            df.at[idx, 'soc'] = current_battery_energy / batt_cap

    # Calculate soc_no_public_charging
    driving_mask = df['occupation'] == 'driving'
    df['soc_no_public_charging'] = np.where(
        driving_mask,
        (df['battery_energy_kwh'] - df['energy_recharged_kwh'].fillna(0)) / batt_cap,
        df['battery_energy_kwh'] / batt_cap
    )

    # Add soc_start (shifted SoC)
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        vehicle_data['soc_start'] = vehicle_data['soc'].shift(1)
        vehicle_data.loc[vehicle_data.index[0], 'soc_start'] = soc_max
        df.loc[vehicle_data.index, 'soc_start'] = vehicle_data['soc_start']

    return df



def evaluate_charging_distribution(df, soc_min, evaluate_per_fleet=True, **kwargs):
    
    # Calculate the number of instances where soc < soc_min for each freight forwarder
    negative_soc_counts = df[df['soc'] < soc_min].groupby('freight_forwarder').size()
    total_counts = df.groupby('freight_forwarder').size()
    negative_soc_percentage = (negative_soc_counts / total_counts) * 100

    if evaluate_per_fleet:
        for ff in negative_soc_counts.index:
            print(f"Freight Forwarder {ff}:")
            print(f"  Instances with soc < soc_min: {negative_soc_counts[ff]}")
            print(f"  Percentage: {negative_soc_percentage[ff]:.2f}% \n")
    
    # Calculate total energy recharged at different locations for reporting
    industrial_recharged = df[(df['occupation'] == 'industrial area') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    home_recharged = df[(df['occupation'] == 'home base') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    public_recharged = df[(df['occupation'] == 'driving') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    total_recharged = df['energy_recharged_kwh'].sum()
    
    print('-- Considering public charging --')
    print(f"Total energy recharged publicly: {public_recharged.sum():0,.2f} kWh")
    print(f"Total energy recharged privately, when using public charging: {(industrial_recharged + home_recharged).sum():0,.2f} kWh")
    print(f"Total energy recharged at home bases, when using public charging: {home_recharged.sum():0,.2f} kWh")
    print(f"Total energy recharged at industrial areas, when using public charging: {industrial_recharged.sum():0,.2f} kWh")
    
    print(f"Total energy recharged publicly (as % of total recharge): {(public_recharged.sum()/total_recharged.sum())*100:0,.2f} %")
    print(f"Total energy recharged at home bases, when using public charging (as % of total recharge): {(home_recharged.sum()/total_recharged.sum())*100:0,.2f} %")
    print(f"Total energy recharged at industrial areas, when using public charging (as % of total recharge): {(industrial_recharged.sum()/total_recharged.sum())*100:0,.2f} %")
    
    return {
            'public': public_recharged.sum() / total_recharged.sum(),
            'destination': industrial_recharged.sum() / total_recharged.sum(),
            'home': home_recharged.sum() / total_recharged.sum()
            }

if __name__ == "__main__":
    # Load only the stops data (which now includes all track information)
    df_stops = pd.read_csv('input/stations/stops.csv', index_col='track_id')
    df_trip_energy= pd.read_csv('output/csvs/tracks_with_energy.csv', index_col='track_id')
    
    df_activities = combine_tracks_and_stops(df_stops, df_trip_energy)
