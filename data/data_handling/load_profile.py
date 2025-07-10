import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def calculate_charging_load_profiles(df_tours, charging_powers, threshold, load_threshold=630, save=False):
    """
    Calculate charging load profiles for each charging station based on tour data.

    Input: DataFrame with energy demand and stop times for each tour
        e.g data/output/track_energies/tours_constant_charging_100-300_no_disp.csv
        
    Parameters:
    -----------
    df_tours : DataFrame
        DataFrame containing tour data with energy demand and stop times
    charging_powers : dict
        Dictionary mapping location types to charging power values
    threshold : float
        Energy threshold in kWh for charging demand calculation
    load_threshold : float, optional
        Threshold in kW to calculate time above threshold, default is 630 kW
    save : bool, optional
        Whether to save the output files, default is False
        
    Returns:
    --------
    cid_load_profiles : dict
        Dictionary of DataFrames, one for each charging station ID (cid),
        containing the charging load profile
    charging_stats : dict
        Dictionary containing statistics for each CID (avg duration, max load, time above threshold)
    """
    df = df_tours.copy()
    
    # Convert stop_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['stop_time']):
        df['stop_time'] = pd.to_datetime(df['stop_time'], utc=True)
    
    # Round stop and start times to the nearest minute
    df['stop_time'] = df['stop_time'].apply(
        lambda x: x - timedelta(seconds=x.second, microseconds=x.microsecond) 
    )

    # Set stop time as index and convert it to local time    
    df.set_index('stop_time', inplace=True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df.reset_index(inplace=True)
    # Extract the date (day) from stop_time
    df['day'] = df['stop_time'].dt.date
    
    # Get unique charging stations (cids)
    unique_cids = df['cid'].unique()
    
    # Dictionary to store results for each cid
    cid_load_profiles = {}
    charging_stats = {}
    cid_freight_forwarders = {}  # Dictionary to track which FF each CID belongs to
    public_energy = 0
    home_base_energy = 0    
    industrial_energy = 0
    
    # Process each charging station separately
    for cid in unique_cids:
        print(f"Processing cid: {cid}")
        # Filter data for current cid
        df_cid = df[df['cid'] == cid].copy()
        
        if len(df_cid) == 0:
            continue
        
        # Get the freight forwarder for this cid
        # Note: This is simplified; in reality, multiple FFs might use the same station
        freight_forwarder = df_cid['freight_forwarder'].iloc[0]
        cid_freight_forwarders[cid] = freight_forwarder
        
        # Find min and max timestamps to create time range
        min_time = df_cid['stop_time'].min()
        max_time = df_cid['stop_time'].max() + pd.Timedelta(hours=10)
        
        # Create a time range with minute-level granularity
        time_range = pd.date_range(start=min_time, end=max_time, freq='1min')
        
        # Create intermediate dataframe for tracking charging packages
        intermediate_df = pd.DataFrame(index=time_range)
        
        # Process each charging session
        for idx, row in df_cid.iterrows():
            arrival = row['stop_time']
            
            # For tracks that require public charging to be completed, assume that the trucks arrives with its min SoC
            energy_demand = min(row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'], threshold)

            if row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'] > threshold:
                public_energy += row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'] - threshold 

            home_base_energy += energy_demand
            industrial_energy += row['energy_recharged_kwh']

            charging_power = charging_powers['home base']
            
            # Calculate charging duration in minutes
            charging_duration_hrs = energy_demand / charging_power
            charging_duration_mins = int(charging_duration_hrs * 60)
            # Keep track of charging durations for calculating average later
            if 'charging_duration_min' not in df_cid:
                df_cid.loc[:, 'charging_duration_min'] = 0
            df_cid.loc[idx, 'charging_duration_min'] = charging_duration_mins
            
            # Calculate the end of charging time
            end_charging_time = arrival + pd.Timedelta(minutes=charging_duration_mins)
            
            # Create a series representing this charging package
            charging_package = pd.Series(
                index=pd.date_range(start=arrival, end=end_charging_time - pd.Timedelta(minutes=1), freq='1min'),
                data=charging_power
            )
            
            # Add this package as a new column to intermediate dataframe
            package_id = f"pkg_{df_cid.loc[idx]['tour_id']}"
            for minute, power in charging_package.items():
                if minute in intermediate_df.index:
                    intermediate_df.loc[minute, package_id] = power

        # Create output dataframe with load profile
        load_profile = pd.DataFrame(index=intermediate_df.index)
        load_profile['date'] = load_profile.index.date
        load_profile['time'] = load_profile.index.time
        
        # Store freight forwarder in load profile
        load_profile['freight_forwarder'] = freight_forwarder
        
        # Calculate total charging load at each minute
        load_profile['load_kW'] = intermediate_df.sum(axis=1)
        
        # Calculate statistics
        max_load = load_profile['load_kW'].max()
        avg_charging_duration = round(df_cid['charging_duration_min'].mean(), 1)
        
        # Calculate time above threshold per day in minutes
        load_profile['above_threshold'] = load_profile['load_kW'] > load_threshold
        # Group by date and sum the minutes above threshold
        above_threshold_per_day = load_profile.groupby('date')['above_threshold'].sum()
        # Calculate average minutes above threshold per day
        avg_minutes_above_threshold = above_threshold_per_day.mean() if not above_threshold_per_day.empty else 0
        # Keep the value in minutes rather than converting to hours
        avg_minutes_above_threshold = round(avg_minutes_above_threshold, 1)
        
        # Store load profile
        cid_load_profiles[cid] = load_profile
        
        # Store statistics
        charging_stats[cid] = {
            'avg_charging_duration_min': avg_charging_duration,
            'max_load_kW': max_load,
            'avg_minutes_above_threshold': avg_minutes_above_threshold,
            'freight_forwarder': freight_forwarder  # Add freight forwarder to statistics
        }
        
        # Print statistics
        print(f"Average charging duration for CID {cid}: {avg_charging_duration:.2f} minutes")
        print(f"Max load for CID {cid}: {max_load:.2f} kW")
        print(f"Average time above {load_threshold} kW per day: {avg_minutes_above_threshold:.1f} minutes")
        print(f"Freight Forwarder: {freight_forwarder}\n")


        if save:
            # Create directory if it doesn't exist
            directory = f"data/output/charging_loads/{charging_powers['home base']}_kW"
            os.makedirs(directory, exist_ok=True)
            # Save the load profile
            load_profile.to_csv(f"{directory}/{cid}_load_profile.csv", index=False)
    
    print('Total energy demand statistics:')
    print(f"Total public energy demand: {public_energy:.2f} kWh")
    print(f"Total home base energy demand: {home_base_energy:.2f} kWh")
    print(f"Total industrial energy demand: {industrial_energy:.2f} kWh")
    print(f"Total energy demand: {public_energy + home_base_energy + industrial_energy:.2f} kWh")
    print(f"public energy demand as percetage of total demand: {(public_energy/(public_energy + home_base_energy + industrial_energy))*100:.2f} %")
    print(f"home base energy demand as percetage of total demand: {(home_base_energy/(public_energy + home_base_energy + industrial_energy))*100:.2f} %")
    print(f"industrial energy demand as percetage of total demand: {(industrial_energy/(public_energy + home_base_energy + industrial_energy))*100:.2f} %")
    
    if save:
        # Convert the nested dictionary to a DataFrame
        stats_df = pd.DataFrame.from_dict(charging_stats, orient='index')
        # Save the statistics
        stats_directory = "data/output/charging_loads"
        os.makedirs(stats_directory, exist_ok=True)
        stats_df.to_csv(f"{stats_directory}/charging_station_stats.csv")
    
    return cid_load_profiles, charging_stats

# Example usage
# load_profiles = calculate_charging_load_profiles('tours_constant_charging_100-300_no_disp.csv')
# First cid's load profile: load_profiles[list(load_profiles.keys())[0]].head()