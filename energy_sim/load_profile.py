import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

cache = joblib.Memory(".cache", verbose=0)

@cache.cache
def calculate_charging_load_profiles(df_trips, charging_power, load_threshold):
    """
    Calculate charging load profiles for each charging station based on tour data.

    Input: DataFrame with energy demand and stop times for each tour
        e.g output/track_energies/tours_constant_charging_100-300_no_disp.csv
        
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
    df = df_trips.loc[df_trips.location == 'home base']
    
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
        
    # Calculate charging power and duration in minutes
    df['charging_duration_min'] = (df['energy_recharged_kwh'] / charging_power * 60).astype(int)
    
    # Calculate charging end time
    df['end_charging_time'] = df['stop_time'] + pd.to_timedelta(df['charging_duration_min'], unit='m')
    
    # Get unique cids and freight forwarders
    cid_freight_forwarders = df.groupby('cid')['freight_forwarder'].first().to_dict()
    
    # Create a DataFrame to hold all charging events expanded by minute
    # For vectorization, create a DataFrame with repeated rows for each minute of charging
    
    # Repeat rows by charging duration
    df_expanded = df.loc[df.index.repeat(df['charging_duration_min'])].copy()
    
    # Create a minute offset for each repeated row
    df_expanded['minute_offset'] = df_expanded.groupby(level=0).cumcount()
    
    # Calculate the timestamp for each minute of charging
    df_expanded['charging_time'] = df_expanded['stop_time'] + pd.to_timedelta(df_expanded['minute_offset'], unit='m')
    
    # Create a multi-index by cid and charging_time
    df_expanded.set_index(['cid', 'charging_time'], inplace=True)
    
    # Assign charging power as load
    df_expanded['load_kW'] = charging_power
    
    # Aggregate load by cid and charging_time
    load_agg = df_expanded.groupby(level=['cid', 'charging_time'])['load_kW'].sum().unstack(level=0).fillna(0)
    
    # Prepare output dictionaries
    cid_load_profiles = {}
    charging_stats = {}
    
    # For each cid, create load profile DataFrame and calculate statistics
    for cid in load_agg.columns:
        load_series = load_agg[cid]
        load_profile = pd.DataFrame({
            'date': load_series.index.date,
            'time': load_series.index.time,
            'freight_forwarder': cid_freight_forwarders.get(cid, None),
            'load_kW': load_series.values
        }, index=load_series.index)
        
        max_load = load_profile['load_kW'].max()
        avg_charging_duration = df.loc[df['cid'] == cid, 'charging_duration_min'].mean()
        
        load_profile['above_threshold'] = load_profile['load_kW'] > load_threshold
        above_threshold_per_day = load_profile.groupby('date')['above_threshold'].sum()
        avg_minutes_above_threshold = above_threshold_per_day.mean() if not above_threshold_per_day.empty else 0
        avg_minutes_above_threshold = round(avg_minutes_above_threshold, 1)
        
        cid_load_profiles[cid] = load_profile
        
        charging_stats[cid] = {
            'avg_charging_duration_min': round(avg_charging_duration, 1) if not pd.isna(avg_charging_duration) else 0,
            'max_load_kW': max_load,
            'avg_minutes_above_threshold': avg_minutes_above_threshold,
            'freight_forwarder': cid_freight_forwarders.get(cid, None)
        }
        
    return cid_load_profiles, charging_stats
