
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib

cache = joblib.Memory(".cache", verbose=0)

@cache.cache
def calculate_charging_load_profiles(df_tours, charging_power, load_threshold, charging_strategy='immediate'):
    """
    Calculate minute-resolution charging load profiles and detailed charging statistics for each charging station (CID).
    Returns both a global DataFrame (all CIDs, minute-based) and a per-CID dictionary of load profiles and stats.

    Parameters
    ----------
    df_tours : pd.DataFrame
        DataFrame with tour data, must include 'stop_time', 'energy_recharged_kwh', 'cid', 'location', 'freight_forwarder', and optionally 'energy_consumption_kwh_cleaned'.
    charging_power : float
        Charging power to use for all home base sessions (kW).
    load_threshold : float
        Power threshold (kW) for statistics (e.g., 630).
    charging_strategy : str, optional
        (Ignored, for compatibility only.)

    Returns
    -------
    load_profile_df : pd.DataFrame
        DataFrame: index = time_ (datetime, minute-resolution), columns = cid, values = load (kW).
    cid_load_profiles : dict
        Dictionary: cid -> DataFrame (minute-based load profile for that CID, with energy breakdown columns).
    charging_stats : dict
        Dictionary: cid -> dict with statistics (max load, avg duration, avg minutes above threshold, energy breakdown, freight forwarder).
    """
    df = df_tours.copy()
    # Only consider home base sessions if 'location' exists
    if 'location' in df.columns:
        df = df[df['location'] == 'home base']
    # Ensure stop_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['stop_time']):
        df['stop_time'] = pd.to_datetime(df['stop_time'], utc=True)
    # Round stop_time to the nearest minute
    df['stop_time'] = df['stop_time'].apply(lambda x: x - timedelta(seconds=x.second, microseconds=x.microsecond))
    # Set stop_time as index and convert to Europe/Berlin
    df.set_index('stop_time', inplace=True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df.reset_index(inplace=True)
    # Extract day
    df['day'] = df['stop_time'].dt.date
    unique_cids = df['cid'].unique()
    charging_stats = {}
    all_charging_entries = []
    cid_load_profiles = {}
    # For energy breakdown
    energy_breakdown = {cid: {'public': 0, 'home_base': 0, 'industrial': 0} for cid in unique_cids}
    # Process each CID separately
    for cid in unique_cids:
        df_cid = df[df['cid'] == cid].copy()
        if len(df_cid) == 0:
            continue
        freight_forwarder = df_cid['freight_forwarder'].iloc[0] if 'freight_forwarder' in df_cid.columns else None
        # Build minute-based load profile for this CID
        charging_minutes = []
        for idx, row in df_cid.iterrows():
            arrival = row['stop_time']
            energy_demand = row['energy_recharged_kwh']
            charging_duration_hrs = energy_demand / charging_power if charging_power > 0 else 0
            charging_duration_mins = int(np.ceil(charging_duration_hrs * 60))
            end_charging_time = arrival + pd.Timedelta(minutes=charging_duration_mins)
            for minute in pd.date_range(start=arrival, end=end_charging_time - pd.Timedelta(minutes=1), freq='1min'):
                charging_minutes.append((minute, charging_power))
                all_charging_entries.append((minute, cid, charging_power))
            # Energy breakdown (if available)
            if 'energy_consumption_kwh_cleaned' in row and not pd.isnull(row['energy_consumption_kwh_cleaned']):
                # Public charging: if energy_consumption > recharged + threshold
                threshold = energy_demand  # fallback if not provided
                if 'energy_consumption_kwh_cleaned' in row:
                    threshold = min(row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'], energy_demand)
                if row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'] > energy_demand:
                    energy_breakdown[cid]['public'] += row['energy_consumption_kwh_cleaned'] - row['energy_recharged_kwh'] - energy_demand
                energy_breakdown[cid]['home_base'] += energy_demand
                # If you have industrial charging, add here if needed
                # energy_breakdown[cid]['industrial'] += ...
            else:
                energy_breakdown[cid]['home_base'] += energy_demand
        # Build per-CID DataFrame
        if charging_minutes:
            cid_profile = pd.DataFrame(charging_minutes, columns=['time_', 'load_kW'])
            cid_profile = cid_profile.groupby('time_').sum()
            cid_profile.index = pd.to_datetime(cid_profile.index)
            cid_profile['date'] = cid_profile.index.date
            cid_profile['time'] = cid_profile.index.time
            cid_profile['freight_forwarder'] = freight_forwarder
            # For stats: max load, avg duration, avg minutes above threshold
            max_load = cid_profile['load_kW'].max()
            avg_charging_duration = np.mean([
                row['energy_recharged_kwh'] / charging_power * 60 if charging_power > 0 else 0
                for _, row in df_cid.iterrows()
            ])
            cid_profile['above_threshold'] = cid_profile['load_kW'] > load_threshold
            above_threshold_per_day = cid_profile.groupby('date')['above_threshold'].sum()
            avg_minutes_above_threshold = above_threshold_per_day.mean() if not above_threshold_per_day.empty else 0
            # Add energy breakdown columns
            for key in energy_breakdown[cid]:
                cid_profile[f'energy_{key}_kWh'] = energy_breakdown[cid][key]
            # Store per-CID profile
            cid_load_profiles[cid] = cid_profile
            # Store stats
            charging_stats[cid] = {
                'avg_charging_duration_min': round(avg_charging_duration, 1),
                'max_load_kW': max_load,
                'avg_minutes_above_threshold': round(avg_minutes_above_threshold, 1),
                'freight_forwarder': freight_forwarder,
                'energy_breakdown': energy_breakdown[cid].copy(),
            }
        else:
            cid_load_profiles[cid] = pd.DataFrame()
            charging_stats[cid] = {
                'avg_charging_duration_min': 0,
                'max_load_kW': 0,
                'avg_minutes_above_threshold': 0,
                'freight_forwarder': freight_forwarder,
                'energy_breakdown': energy_breakdown[cid].copy(),
            }
    # Build global minute-based load profile DataFrame (all CIDs)
    if all_charging_entries:
        df_charging = pd.DataFrame(all_charging_entries, columns=['time_', 'cid', 'power'])
        load_profile_df = df_charging.pivot_table(index='time_', columns='cid', values='power', aggfunc='sum').fillna(0)
        load_profile_df.index.name = 'time_'
    else:
        load_profile_df = pd.DataFrame()
    return load_profile_df, cid_load_profiles, charging_stats

