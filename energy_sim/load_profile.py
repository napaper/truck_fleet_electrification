"""
Charging load profile generation for truck fleet electrification analysis.

This module provides functions to calculate minute-resolution charging load profiles
and detailed charging statistics for each charging station (CID). It processes tour
data to generate comprehensive load profiles that can be used for infrastructure
planning and grid impact analysis.

The module supports different charging strategies and provides energy breakdown
analysis between home base, public, and industrial charging sources.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import joblib

# Initialize caching for performance optimization
cache = joblib.Memory(".cache", verbose=0)


@cache.cache
def calculate_charging_load_profiles(df_tours, charging_power, load_threshold, charging_strategy='immediate'):
    """
    Calculate minute-resolution charging load profiles and detailed charging statistics.
    
    This function processes tour data to generate comprehensive charging load profiles
    for each charging station (CID), including minute-by-minute power demand and
    detailed statistics for infrastructure planning.
    
    Args:
        df_tours (pd.DataFrame): DataFrame with tour data, must include:
            - 'stop_time': Arrival time at charging station
            - 'energy_recharged_kwh': Energy demand for recharging
            - 'cid': Charging station identifier
            - 'location': Charging location type
            - 'freight_forwarder': Freight forwarder identifier
            - 'energy_consumption_kwh_cleaned': Optional, for energy breakdown analysis
        charging_power (float): Charging power to use for all home base sessions (kW)
        load_threshold (float): Power threshold (kW) for statistics calculation
        charging_strategy (str, optional): Charging strategy identifier (for compatibility)
        # TODO: Implement charging strategy logic in future versions:
        # - 'immediate': Charge immediately upon arrival (current default)
        # - 'delayed': Delay charging to off-peak hours
        # - 'smart': Optimize charging based on grid load and energy prices
        # - 'scheduled': Pre-scheduled charging windows
            
    Returns:
        tuple: Contains three elements:
            - load_profile_df (pd.DataFrame): Global minute-resolution load profile
                with time index and CID columns, values represent power demand (kW)
            - cid_load_profiles (dict): Per-CID load profiles with detailed breakdown
            - charging_stats (dict): Comprehensive statistics for each CID including
                average duration, maximum load, and energy breakdown
                
    Note:
        The function automatically filters for home base charging sessions and
        converts all timestamps to Europe/Berlin timezone for consistency.
        Results are cached using joblib for performance optimization.
    """
    # Create a copy to avoid modifying the original data
    df = df_tours.copy()
    
    # Filter for home base charging sessions only
    if 'location' in df.columns:
        df = df[df['location'] == 'home base']
    
    # Ensure stop_time is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['stop_time']):
        df['stop_time'] = pd.to_datetime(df['stop_time'], utc=True)
    
    # Round stop_time to the nearest minute for consistent time resolution
    df['stop_time'] = df['stop_time'].apply(
        lambda x: x - timedelta(seconds=x.second, microseconds=x.microsecond)
    )
    
    # Convert timezone to Europe/Berlin and extract date information
    df.set_index('stop_time', inplace=True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df.reset_index(inplace=True)
    df['day'] = df['stop_time'].dt.date
    
    # Initialize data structures for results
    unique_cids = df['cid'].unique()
    charging_stats = {}
    all_charging_entries = []
    cid_load_profiles = {}
    
    # Initialize energy breakdown tracking for each CID
    energy_breakdown = {
        cid: {'public': 0, 'home_base': 0, 'industrial': 0} 
        for cid in unique_cids
    }
    
    # Process each charging station (CID) separately
    for cid in unique_cids:
        df_cid = df[df['cid'] == cid].copy()
        
        # Skip empty CID datasets
        if len(df_cid) == 0:
            continue
            
        # Extract freight forwarder information for this CID
        freight_forwarder = df_cid['freight_forwarder'].iloc[0] if 'freight_forwarder' in df_cid.columns else None
        
        # Build minute-based load profile for this CID
        charging_minutes = []
        
        # Process each charging session for the current CID
        for idx, row in df_cid.iterrows():
            arrival = row['stop_time']
            energy_demand = row['energy_recharged_kwh']
            
            # Calculate charging duration based on power and energy demand
            charging_duration_hrs = energy_demand / charging_power if charging_power > 0 else 0
            charging_duration_mins = int(np.ceil(charging_duration_hrs * 60))
            end_charging_time = arrival + pd.Timedelta(minutes=charging_duration_mins)
            
            # Generate minute-by-minute charging entries
            for minute in pd.date_range(
                start=arrival, 
                end=end_charging_time - pd.Timedelta(minutes=1), 
                freq='1min'
            ):
                charging_minutes.append((minute, charging_power))
                all_charging_entries.append((minute, cid, charging_power))
            
            # Analyze energy breakdown if consumption data is available
            if 'energy_consumption_kwh_cleaned' in row and not pd.isnull(row['energy_consumption_kwh_cleaned']):
                # Calculate energy that exceeds home base charging capacity
                # This represents energy that would need public charging
                threshold = energy_demand  # fallback threshold
                if 'energy_consumption_kwh_cleaned' in row:
                    threshold = min(
                        row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'], 
                        energy_demand
                    )
                
                # Identify public charging needs
                if row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'] > energy_demand:
                    energy_breakdown[cid]['public'] += (
                        row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'] - energy_demand
                    )
                
                # Record home base charging energy
                energy_breakdown[cid]['home_base'] += energy_demand
                
                # Note: Industrial charging can be added here if needed
                # energy_breakdown[cid]['industrial'] += ...
            else:
                # If no consumption data available, assume all energy is home base charging
                energy_breakdown[cid]['home_base'] += energy_demand
        
        # Build comprehensive per-CID DataFrame with load profile and statistics
        if charging_minutes:
            # Create minute-resolution load profile
            cid_profile = pd.DataFrame(charging_minutes, columns=['time_', 'load_kW'])
            cid_profile = cid_profile.groupby('time_').sum()
            cid_profile.index = pd.to_datetime(cid_profile.index)
            
            # Add metadata columns for analysis
            cid_profile['date'] = cid_profile.index.date
            cid_profile['time'] = cid_profile.index.time
            cid_profile['freight_forwarder'] = freight_forwarder
            
            # Calculate key statistics for this CID
            max_load = cid_profile['load_kW'].max()
            avg_charging_duration = np.mean([
                row['energy_recharged_kwh'] / charging_power * 60 if charging_power > 0 else 0
                for _, row in df_cid.iterrows()
            ])
            
            # Analyze load threshold exceedance
            cid_profile['above_threshold'] = cid_profile['load_kW'] > load_threshold
            above_threshold_per_day = cid_profile.groupby('date')['above_threshold'].sum()
            avg_minutes_above_threshold = above_threshold_per_day.mean() if not above_threshold_per_day.empty else 0
            
            # Add energy breakdown columns to the profile
            for key in energy_breakdown[cid]:
                cid_profile[f'energy_{key}_kWh'] = energy_breakdown[cid][key]
            
            # Store the complete CID profile
            cid_load_profiles[cid] = cid_profile
            
            # Compile comprehensive statistics for this CID
            charging_stats[cid] = {
                'avg_charging_duration_min': round(avg_charging_duration, 1),
                'max_load_kW': max_load,
                'avg_minutes_above_threshold': round(avg_minutes_above_threshold, 1),
                'freight_forwarder': freight_forwarder,
                'energy_breakdown': energy_breakdown[cid].copy(),
            }
        else:
            # Handle case where no charging minutes were generated
            cid_load_profiles[cid] = pd.DataFrame()
            charging_stats[cid] = {
                'avg_charging_duration_min': 0,
                'max_load_kW': 0,
                'avg_minutes_above_threshold': 0,
                'freight_forwarder': freight_forwarder,
                'energy_breakdown': energy_breakdown[cid].copy(),
            }
    
    # Build global minute-based load profile DataFrame (all CIDs combined)
    if all_charging_entries:
        df_charging = pd.DataFrame(all_charging_entries, columns=['time_', 'cid', 'power'])
        load_profile_df = df_charging.pivot_table(
            index='time_', 
            columns='cid', 
            values='power', 
            aggfunc='sum'
        ).fillna(0)
        load_profile_df.index.name = 'time_'
    else:
        # Return empty DataFrame if no charging entries exist
        load_profile_df = pd.DataFrame()
    
    return load_profile_df, cid_load_profiles, charging_stats

