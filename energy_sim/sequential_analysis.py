"""
Sequential analysis functions for truck fleet electrification analysis.

This module provides functions for analyzing truck activities sequentially over time,
including combining track and stop data, calculating battery state of charge (SoC),
and evaluating charging distribution patterns across different locations.

The module supports intelligent charging strategies that optimize battery usage
while ensuring vehicles can reach their destinations without running out of energy.
"""

import pandas as pd
import numpy as np
import joblib

# Initialize caching for performance optimization
cache = joblib.Memory(".cache")  # , verbose=0


# =============================================================================
# DATA COMBINATION AND ACTIVITY CREATION
# =============================================================================

@cache.cache
def combine_tracks_and_stops(df_stops, df_tracks_with_energy):
    """
    Combine track and stop data to create alternating driving and rest activities.
    
    This function processes stops data which contains both track (driving) and stop
    (rest) information, creating a comprehensive timeline of vehicle activities.
    It ensures that all activities include the charging station identifier (cid)
    and sets the last occupation of each tour to 'home base'.
    
    Args:
        df_stops (pd.DataFrame): DataFrame containing both track and stop information
            with 'location' column for stops and other activity details
        df_tracks_with_energy (pd.DataFrame, optional): Energy consumption data
            for tracks. Not used in this version as df_stops contains all necessary
            information, but kept for compatibility
            
    Returns:
        pd.DataFrame: Combined dataframe with alternating driving and rest periods,
            sorted by vehicle_id and start_time, with proper occupation labeling
            
    Note:
        The function processes each vehicle separately to maintain chronological
        order and creates rest activities between driving segments based on
        location information and timing data.
    """
    # Create a copy to avoid modifying the original data
    df_stops = df_stops.copy()

    # Add energy consumption data if provided (for compatibility)
    if df_tracks_with_energy is not None:
        df_stops['energy_consumption_kwh_cleaned'] = df_tracks_with_energy['energy_consumption_kwh_cleaned']
        df_stops['avg_energy_consumption_kwh/km_cleaned'] = df_tracks_with_energy['avg_energy_consumption_kwh/km_cleaned']
    
    # Initialize list to store all activities (driving and rests)
    all_activities = []
    
    # Get unique vehicle IDs for sequential processing
    vehicles = df_stops['vehicle_id'].unique()
    
    # Process each vehicle separately to maintain chronological order
    for vehicle_id in vehicles:
        print(f'Processing Vehicle ID: {vehicle_id}')
        
        # Get all stops for this vehicle, sorted by start time
        vehicle_data = df_stops[df_stops['vehicle_id'] == vehicle_id].sort_values('start_time')
        
        # Skip vehicles with no data
        if len(vehicle_data) == 0:
            continue
        
        # Process each entry chronologically
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
            
            # Create rest activity if location information is available
            if 'location' in entry and pd.notna(entry['location']):
                # Use the location as the occupation for the rest period
                occupation = entry['location']
                
                # Determine rest period end time based on next activity
                if i < len(vehicle_data) - 1 and pd.notna(vehicle_data.iloc[i + 1]['start_time']):
                    rest_end_time = vehicle_data.iloc[i + 1]['start_time']
                    
                    # Create rest activity with required fields
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
                    
                    # Ensure cid is included in rest activities
                    rest_activity_dict['cid'] = entry['cid']
                    
                    rest_activity = pd.Series(rest_activity_dict)
                    all_activities.append(rest_activity)
                    
                    # Store this as the last activity for this tour_id
                    tour_id = entry['tour_id']
    
    # Convert the list of activities to a DataFrame
    df_activities = pd.DataFrame(all_activities)

    # Ensure tour_id is numeric for proper sorting and grouping
    df_activities['tour_id'] = pd.to_numeric(df_activities['tour_id'], downcast='integer')
    
    # Sort by vehicle_id and start_time for chronological order
    df_activities = df_activities.sort_values(['vehicle_id', 'start_time'])
    
    # Remove any duplicate columns that may have been created
    df_activities = df_activities.loc[:, ~df_activities.columns.duplicated()]
    
    # Override the occupation of the last activity of each tour_id with 'home base'
    # This ensures all tours end at the home base for consistency
    last_activities = df_activities.groupby('tour_id').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    
    # Set the occupation to 'home base' for the last activity of each tour
    for idx in last_activities.index:
        tour_id = last_activities.loc[idx, 'tour_id']
        # Find all rows with this tour_id
        tour_activities = df_activities[df_activities['tour_id'] == tour_id]
        # Get the index of the last activity (latest stop_time)
        last_idx = tour_activities[tour_activities.stop_time == tour_activities.stop_time.max()].index
        # Set the occupation to 'home base'
        df_activities.loc[last_idx, 'occupation'] = 'home base'

    # Save the combined dataframe to a CSV file (commented out by default)
    # df_activities.to_csv('output/csvs/tracks_and_stops.csv', index=False)
    
    # Reset index for clean DataFrame structure
    df_activities = df_activities.reset_index(drop=True)
    
    return df_activities


# =============================================================================
# BATTERY STATE OF CHARGE CALCULATION
# =============================================================================

@cache.cache
def truck_soc(df_activities, charging_powers, batt_cap, soc_min, soc_max, **kwargs):
    """
    Calculate battery state of charge (SoC) for each vehicle activity.
    
    This function simulates the battery energy levels throughout the day for each
    vehicle, considering energy consumption during driving and energy addition
    during charging stops. It implements intelligent charging strategies that
    optimize battery usage while ensuring vehicles can reach their destinations.
    
    Args:
        df_activities (pd.DataFrame): DataFrame with vehicle activities including
            driving and charging periods, sorted by vehicle_id and start_time
        charging_powers (dict): Dictionary mapping occupation types to charging
            power levels (kW) for different charging locations
        batt_cap (float): Battery capacity in kWh
        soc_min (float): Minimum allowed state of charge (0.0 to 1.0)
        soc_max (float): Maximum allowed state of charge (0.0 to 1.0)
        **kwargs: Additional keyword arguments for future extensions
            
    Returns:
        pd.DataFrame: Updated activities DataFrame with additional columns:
            - 'energy_recharged_kwh': Energy added during charging
            - 'battery_energy_kwh': Current battery energy level
            - 'soc': State of charge as percentage of capacity
            - 'soc_no_public_charging': SoC without public charging
            - 'soc_start': SoC at the start of each activity
            
    Note:
        The function implements intelligent charging strategies that defer charging
        when safe to do so, optimizing for home base charging while ensuring
        vehicles never drop below the minimum SoC threshold.
    """
    import numpy as np
    import pandas as pd

    # Create a copy and ensure proper sorting
    df = df_activities.copy()
    df = df.sort_values(['vehicle_id', 'start_time'])

    # Initialize energy and SoC columns
    df['energy_recharged_kwh'] = None     
    df['battery_energy_kwh'] = None
    df['soc'] = None

    # Calculate energy limits based on SoC constraints
    max_battery_energy = batt_cap * soc_max
    min_battery_energy = batt_cap * soc_min

    # Process each vehicle sequentially
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        current_battery_energy = max_battery_energy  # Start with full battery

        # Process each activity for the current vehicle
        for idx in vehicle_data.index:
            activity = vehicle_data.loc[idx]
            energy_added = 0

            # Handle driving activities (energy consumption)
            if activity['occupation'] == 'driving':
                if 'energy_consumption_kwh_cleaned' in df.columns and pd.notna(activity['energy_consumption_kwh_cleaned']):
                    current_battery_energy -= activity['energy_consumption_kwh_cleaned']
                    # Emergency top-up to prevent dropping below minimum SoC
                    if current_battery_energy < min_battery_energy:
                        energy_needed = min_battery_energy - current_battery_energy
                        current_battery_energy = min_battery_energy
                        energy_added = energy_needed  # Track emergency charging

            # Handle charging activities (energy addition)
            else:
                occupation = activity['occupation']
                if occupation in charging_powers:
                    charging_power = charging_powers[occupation]
                    charging_time = activity['duration_h']
                    energy_possible_now = charging_power * charging_time
                    energy_to_full = max_battery_energy - current_battery_energy

                    # Implement intelligent charging strategy for non-home base locations
                    if occupation != 'home base' and occupation != 'driving':
                        # Check whether we can safely skip charging and reach the next home base
                        future_activities = vehicle_data.loc[idx+1:]
                        cumulative_consumption = 0
                        home_found = False

                        # Calculate cumulative energy consumption until next home base
                        for future_idx, future_activity in future_activities.iterrows():
                            if future_activity['occupation'] == 'driving' and pd.notna(future_activity.get('energy_consumption_kwh_cleaned', None)):
                                cumulative_consumption += future_activity['energy_consumption_kwh_cleaned']
                            elif future_activity['occupation'] == 'home base':
                                home_found = True
                                break

                        # Decide whether to charge now or defer
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

                    # Enforce battery energy bounds with small tolerance for floating point errors
                    assert current_battery_energy >= min_battery_energy - 1e-3, (
                        f"Energy dropped below min: {current_battery_energy:.2f} < {min_battery_energy:.2f}"
                    )
                    assert current_battery_energy <= max_battery_energy + 1e-3, (
                        f"Energy exceeded max: {current_battery_energy:.2f} > {max_battery_energy:.2f}"
                    )

            # Update the DataFrame with calculated values
            df.at[idx, 'battery_energy_kwh'] = current_battery_energy
            df.at[idx, 'energy_recharged_kwh'] = energy_added
            df.at[idx, 'soc'] = current_battery_energy / batt_cap

    # Calculate SoC without public charging for comparison analysis
    driving_mask = df['occupation'] == 'driving'
    df['soc_no_public_charging'] = np.where(
        driving_mask,
        (df['battery_energy_kwh'] - df['energy_recharged_kwh'].fillna(0)) / batt_cap,
        df['battery_energy_kwh'] / batt_cap
    )

    # Add soc_start (SoC at the beginning of each activity) for analysis
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        vehicle_data['soc_start'] = vehicle_data['soc'].shift(1)
        vehicle_data.loc[vehicle_data.index[0], 'soc_start'] = soc_max  # First activity starts with full battery
        df.loc[vehicle_data.index, 'soc_start'] = vehicle_data['soc_start']

    return df


# =============================================================================
# CHARGING DISTRIBUTION EVALUATION
# =============================================================================

def evaluate_charging_distribution(df, soc_min, evaluate_per_fleet=True, **kwargs):
    """
    Evaluate charging distribution patterns across different locations and fleets.
    
    This function analyzes the distribution of energy recharging across different
    charging locations (home base, industrial areas, public charging) and provides
    detailed statistics about SoC violations and energy distribution patterns.
    
    Args:
        df (pd.DataFrame): Activities DataFrame with SoC and energy recharging data
        soc_min (float): Minimum allowed state of charge threshold
        evaluate_per_fleet (bool): Whether to print detailed per-fleet statistics
        **kwargs: Additional keyword arguments for future extensions
            
    Returns:
        dict: Dictionary containing charging distribution percentages:
            - 'public': Percentage of energy recharged at public locations
            - 'destination': Percentage of energy recharged at industrial areas
            - 'home': Percentage of energy recharged at home bases
            
    Note:
        The function provides comprehensive reporting on charging patterns and
        identifies potential issues with SoC violations across different fleets.
    """
    
    # Calculate SoC violations for each freight forwarder
    negative_soc_counts = df[df['soc'] < soc_min].groupby('freight_forwarder').size()
    total_counts = df.groupby('freight_forwarder').size()
    negative_soc_percentage = (negative_soc_counts / total_counts) * 100

    # Print detailed per-fleet statistics if requested
    if evaluate_per_fleet:
        for ff in negative_soc_counts.index:
            print(f"Freight Forwarder {ff}:")
            print(f"  Instances with soc < soc_min: {negative_soc_counts[ff]}")
            print(f"  Percentage: {negative_soc_percentage[ff]:.2f}% \n")
    
    # Calculate total energy recharged at different locations for comprehensive reporting
    industrial_recharged = df[(df['occupation'] == 'industrial area') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    home_recharged = df[(df['occupation'] == 'home base') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    public_recharged = df[(df['occupation'] == 'driving') & df['energy_recharged_kwh'].notna()]['energy_recharged_kwh'].sum()
    total_recharged = df['energy_recharged_kwh'].sum()
    
    # Print comprehensive charging distribution report
    print('-- Charging Distribution Analysis (with public charging) --')
    print(f"Total energy recharged publicly: {public_recharged.sum():0,.2f} kWh")
    print(f"Total energy recharged privately: {(industrial_recharged + home_recharged).sum():0,.2f} kWh")
    print(f"Total energy recharged at home bases: {home_recharged.sum():0,.2f} kWh")
    print(f"Total energy recharged at industrial areas: {industrial_recharged.sum():0,.2f} kWh")
    
    # Calculate and display percentage breakdowns
    print(f"Public charging (as % of total): {(public_recharged.sum()/total_recharged.sum())*100:0,.2f} %")
    print(f"Home base charging (as % of total): {(home_recharged.sum()/total_recharged.sum())*100:0,.2f} %")
    print(f"Industrial area charging (as % of total): {(industrial_recharged.sum()/total_recharged.sum())*100:0,.2f} %")
    
    # Return charging distribution percentages for further analysis
    return {
        'public': public_recharged.sum() / total_recharged.sum(),
        'destination': industrial_recharged.sum() / total_recharged.sum(),
        'home': home_recharged.sum() / total_recharged.sum()
    }


# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Load data for testing and demonstration
    df_stops = pd.read_csv('input/stations/stops.csv', index_col='track_id')
    df_trip_energy = pd.read_csv('output/csvs/tracks_with_energy.csv', index_col='track_id')
    
    # Execute the main analysis pipeline
    df_activities = combine_tracks_and_stops(df_stops, df_trip_energy)
