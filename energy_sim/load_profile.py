import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

cache = joblib.Memory(".cache", verbose=0)


@cache.cache
def calculate_charging_load_profiles(df_, charging_power, load_threshold, charging_strategy='immediate'):
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
    load_threshold : float, optional
        Threshold in kW to calculate time above threshold, default is 630 kW
    charging_strategy : str, optional
        Charging strategy to use: 'immediate' (default), 'delayed', or 'average'
        
    Returns:
    --------
    cid_load_profiles : dict
        Dictionary of DataFrames, one for each charging station ID (cid),
        containing the charging load profile
    charging_stats : dict
        Dictionary containing statistics for each CID (avg duration, max load, time above threshold)
    """
    df = df_.loc[(df_.location == 'home base') & (df_.distance.isna())]
    
    # Round stop and start times to the nearest minute
    df = df.assign(stop_time_=df['stop_time'].dt.floor('min'))
  
    if charging_strategy == 'delayed':
        # Adjust start charging time to be as late as possible before stop_time
        df = df.assign(charging_duration_min=(df['energy_recharged_kwh'] / charging_power * 60).astype(int))
        df = df.assign(
            start_charging_time=df['stop_time_'] + pd.to_timedelta((df['duration'] / 60 - df['charging_duration_min']).fillna(0).astype(int), unit='m')
        )
        df = df.assign(charging_power=charging_power)
    elif charging_strategy == 'average':
        # Charge evenly over the whole time frame from start_charging_time to stop_time_
        # Here, we assume start_charging_time is the earliest time charging can start, e.g. arrival time
        # For simplicity, assume charging window is from (stop_time_ - charging_duration_min) to stop_time_
        # So average power is charging_power spread evenly over charging_duration_min
        df = df.assign(charging_duration_min=(df['duration'] / 60).astype(int))
        df = df.assign(start_charging_time=df['stop_time_'])
        df = df.assign(charging_power=df['energy_recharged_kwh'] / df['charging_duration_min'] * 60)
    elif charging_strategy == 'immediate':
        # Immediate charging: start charging at stop_time_
        df = df.assign(charging_duration_min=(df['energy_recharged_kwh'] / charging_power * 60).astype(int))
        df = df.assign(start_charging_time=df['stop_time_'])
        df = df.assign(charging_power=charging_power)
    else:
        raise ValueError("Invalid charging strategy. Choose from 'immediate', 'delayed', or 'average'.")
    
    df = df.assign(end_charging_time=df['start_charging_time'] + pd.to_timedelta(df['charging_duration_min'], unit='m'))
    
    # Expand df to one row per minute of charging using the charging_duration_min column
    df_expanded = df.loc[df.index.repeat(df['charging_duration_min'])].copy()
    df_expanded['minute_offset'] = df_expanded.groupby(level=0).cumcount()

    df_expanded['time_'] = df_expanded['start_charging_time'] + pd.to_timedelta(df_expanded['minute_offset'], unit='m')
    
    load_agg = df_expanded.groupby(['cid', 'time_']).charging_power.sum().unstack(level=0).fillna(0)
    
    # Prepare output dictionaries
    cid_load_profiles = {}
    charging_stats = {}
   
    # For each cid, create load profile DataFrame and calculate statistics
    charging_stats = {
        'max_load_kW': load_agg.max().to_dict(),
        'minutes_above_threshold': (load_agg > load_threshold).sum().to_dict(),
    }
        
    return load_agg, charging_stats
