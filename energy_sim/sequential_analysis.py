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
    df_activities.to_csv('output/csvs/tracks_and_stops.csv', index=False)
    
    # Reset index
    df_activities = df_activities.reset_index(drop=True)
    
    return df_activities


@cache.cache
def truck_soc(df_activities, charging_powers, soc_min=0.15):
    """
    Adds battery energy, state of charge (SoC), and energy recharged columns to the activities dataframe.
    
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
        DataFrame with added battery_energy_kwh, energy_recharged_kwh and soc columns
    """
    # Create a copy to avoid modifying the original
    df = df_activities.copy()
    
    # Sort by vehicle_id and start_time to ensure proper sequence
    df = df.sort_values(['vehicle_id', 'start_time'])
    
    # Add battery_energy, soc, and energy_recharged columns
    df['energy_recharged_kwh'] = None     
    df['battery_energy_kwh'] = None
    df['soc'] = None
    

    # Define battery capacity
    battery_capacity = 572  # kWh
    max_soc = 0.9  # Maximum state of charge (90%)
    max_battery_energy = battery_capacity * max_soc
    min_battery_energy = battery_capacity * soc_min
    
    # Create an empty dataframe to record emergency charging events
    public_charging = pd.DataFrame(
        columns=['freight_forwarder', 'public_energy', 'private energy'],
        index=pd.Index([], name='vehicle_id'))

    
    # Process each vehicle separately
    for vehicle_id in df['vehicle_id'].unique():
        # Get all activities for this vehicle
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        
        # Initialize battery energy for first activity
        current_battery_energy = max_battery_energy  # Start at 90% SoC
        current_battery_energy_w_pub = max_battery_energy

        # Get freight_forwarder value for this vehicle
        freight_forwarder = vehicle_data['freight_forwarder'].iloc[0] if 'freight_forwarder' in vehicle_data.columns and not vehicle_data['freight_forwarder'].isna().all() else None
        
        # Add a row to emergency_charging_events
        # Create a new DataFrame with vehicle_id as index
        new_entry = pd.DataFrame({
            'freight_forwarder': [freight_forwarder],
            'public_energy': [0],
            'home base energy': [0], 
            'industrial area energy': [0], 
        }, index=[vehicle_id])
        # Set the index name
        new_entry.index.name = 'vehicle_id'
        # Concatenate with existing DataFrame
        public_charging = pd.concat([public_charging, new_entry])
        
        # Process each activity in sequence
        for idx in vehicle_data.index:
            activity = vehicle_data.loc[idx]
            
            if activity['occupation'] == 'driving':
                # Subtract energy consumption for driving
                if 'energy_consumption_kwh_cleaned' in df.columns and pd.notna(activity['energy_consumption_kwh_cleaned']):
                    current_battery_energy -= activity['energy_consumption_kwh_cleaned']
                    current_battery_energy_w_pub -= activity['energy_consumption_kwh_cleaned']
                    current_battery_energy = min(current_battery_energy, max_battery_energy)
                    current_battery_energy_w_pub = min(current_battery_energy_w_pub, max_battery_energy)
                    if current_battery_energy_w_pub < min_battery_energy:
                        req_charge = min_battery_energy - current_battery_energy_w_pub
                        current_battery_energy_w_pub = min_battery_energy
                        public_charging.loc[vehicle_id, 'public_energy'] += req_charge


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
                    energy_added = min(energy_charged, max_battery_energy - max(current_battery_energy, battery_capacity*soc_min))
                    
                    energy_charged_w_pub = min(energy_charged, max_battery_energy - current_battery_energy_w_pub)
                    
                    if current_battery_energy_w_pub < min_battery_energy or energy_charged_w_pub > max_battery_energy:
                        raise ValueError(f"Invalid energy values: current_battery_energy_w_pub is below minimum or energy_charged_w_pub exceeds maximum")
                    
                    # Add charged energy, but don't exceed max battery energy
                    current_battery_energy = min(current_battery_energy + energy_charged, max_battery_energy)

                    current_battery_energy_w_pub = current_battery_energy_w_pub + energy_charged_w_pub
                    public_charging.loc[vehicle_id, f'{occupation} energy'] += energy_charged_w_pub

                    if current_battery_energy_w_pub < current_battery_energy:
                        raise ValueError(f"Invalid energy values: current battery energy with public charging should not be less then without public charging")

                    # Record recharged energy (consolidated into a single column)
                    if energy_added > 0:
                        df.at[idx, 'energy_recharged_kwh'] = energy_added

                df.at[idx, 'battery_energy_kwh'] = current_battery_energy
            
            # Calculate SoC
            df.at[idx, 'soc'] = current_battery_energy / battery_capacity
    
    for vehicle_id in df['vehicle_id'].unique():
        # Get all activities for this vehicle
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        vehicle_data['soc_start'] = vehicle_data['soc'].shift(1)
        vehicle_data.loc[vehicle_data.index[0], 'soc_start'] = max_soc  # Set max_soc for the first activity
        # Update the main dataframe
        df.loc[vehicle_data.index, 'soc_start'] = vehicle_data['soc_start']      


    # Round to reasonable precision
    df['battery_energy_kwh'] = pd.to_numeric(df['battery_energy_kwh'])
    df['energy_recharged_kwh'] = pd.to_numeric(df['energy_recharged_kwh'])
    df['soc'] = pd.to_numeric(df['soc'])
    df['battery_energy_kwh'] = df['battery_energy_kwh'].round(2)
    
    # Only round where df it's not None
    mask = df['energy_recharged_kwh'].notna()
    if mask.any():
        df.loc[mask, 'energy_recharged_kwh'] = df.loc[mask, 'energy_recharged_kwh'].round(2)

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
    
    # Calculate total energy recharged at different locations for reporting
    industrial_recharged = df[(df['occupation'] == 'industrial area') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    home_recharged = df[(df['occupation'] == 'home base') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    total_recharged = df['energy_recharged_kwh'].sum()
    
    print('Without considering public charging:')
    print(f"Energy recharged at industrial areas: {industrial_recharged:.2f} kWh")
    print(f"Energy recharged at home bases: {home_recharged:.2f} kWh")
    print(f"Total energy recharged at private areas: {total_recharged:.2f} kWh")
    print('\n')

    total_recharged_publicly = public_charging['public_energy'].sum()
    total_recharged_indust = public_charging['industrial area energy'].sum()
    total_recharged_home = public_charging['home base energy'].sum()
    total_recharged_private = total_recharged_indust + total_recharged_home

    print('Considering public charging:')
    print(f"Total energy recharged publicly: {total_recharged_publicly:.2f} kWh")
    print(f"Total energy recharged privately, when using public charging: {total_recharged_private:.2f} kWh")
    print(f"Total energy recharged at home bases, when using public charging: {total_recharged_home:.2f} kWh")
    print(f"Total energy recharged at industrial areas, when using public charging: {total_recharged_indust:.2f} kWh")
    print(f"Total energy recharged publicly (as % of total recharge): {(total_recharged_publicly/(total_recharged_publicly + total_recharged_private))*100:.2f} %")
    print(f"Total energy recharged at home bases, when using public charging (as % of total recharge): {(total_recharged_home/(total_recharged_publicly + total_recharged_private))*100:.2f} %")
    print(f"Total energy recharged at industrial areas, when using public charging (as % of total recharge): {(total_recharged_indust/(total_recharged_publicly + total_recharged_private))*100:.2f} %")

    # Save the updated dataframe to a CSV file
    df.to_csv(f"output/truck_socs/activities_constant_charging_{charging_powers['home base']}-{charging_powers['industrial area']}.csv", index=False)
    
    return df, public_charging

if __name__ == "__main__":
    # Load only the stops data (which now includes all track information)
    df_stops = pd.read_csv('input/stations/stops.csv', index_col='track_id')
    df_trip_energy= pd.read_csv('output/csvs/tracks_with_energy.csv', index_col='track_id')
    
    df_activities = combine_tracks_and_stops(df_stops, df_trip_energy)