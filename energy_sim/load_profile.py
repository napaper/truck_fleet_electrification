
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

cache = joblib.Memory(".cache", verbose=0)

@cache.cache
def calculate_charging_load_profiles(df_tours, charging_power, load_threshold, charging_strategy='immediate'):
    """
    Calculate charging load profiles for each charging station based on tour data.

    Parameters:
    -----------
    df_tours : DataFrame
        DataFrame containing tour data with energy demand and stop times
    charging_power : float
        Charging power value for home base (kW)
    load_threshold : float
        Threshold in kW to calculate time above threshold, default is 630 kW
    charging_strategy : str, optional
        (Wird ignoriert, nur für Kompatibilität)
    Returns:
    --------
    load_profile_df : DataFrame
        DataFrame: index = time (datetime), columns = cid, values = load (kW)
    charging_stats : dict
        Dictionary containing statistics for each CID (avg duration, max load, time above threshold)
    """
    df = df_tours.copy()
    # Nur home base berücksichtigen, falls Spalte location existiert
    if 'location' in df.columns:
        df = df[df['location'] == 'home base']
    # Konvertiere stop_time zu datetime falls nötig
    if not pd.api.types.is_datetime64_any_dtype(df['stop_time']):
        df['stop_time'] = pd.to_datetime(df['stop_time'], utc=True)
    # Runde stop_time auf die nächste Minute
    df['stop_time'] = df['stop_time'].apply(lambda x: x - timedelta(seconds=x.second, microseconds=x.microsecond))
    # Setze stop_time als Index und konvertiere zu Europe/Berlin
    df.set_index('stop_time', inplace=True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df.reset_index(inplace=True)
    # Extrahiere Tag
    df['day'] = df['stop_time'].dt.date
    unique_cids = df['cid'].unique()
    charging_stats = {}
    all_charging_entries = []
    for cid in unique_cids:
        df_cid = df[df['cid'] == cid].copy()
        if len(df_cid) == 0:
            continue
        # Jede Ladesession
        for idx, row in df_cid.iterrows():
            arrival = row['stop_time']
            energy_demand = row['energy_recharged_kwh']
            charging_duration_hrs = energy_demand / charging_power if charging_power > 0 else 0
            charging_duration_mins = int(np.ceil(charging_duration_hrs * 60))
            end_charging_time = arrival + pd.Timedelta(minutes=charging_duration_mins)
            for minute in pd.date_range(start=arrival, end=end_charging_time - pd.Timedelta(minutes=1), freq='1min'):
                all_charging_entries.append((minute, cid, charging_power))
        # Statistiken
        max_load = 0
        avg_charging_duration = 0
        avg_minutes_above_threshold = 0
        if len(df_cid) > 0:
            avg_charging_duration = np.mean([
                row['energy_recharged_kwh'] / charging_power * 60 if charging_power > 0 else 0
                for _, row in df_cid.iterrows()
            ])
        charging_stats[cid] = {
            'avg_charging_duration_min': round(avg_charging_duration, 1),
            'max_load_kW': None,  # wird nachher gesetzt
            'avg_minutes_above_threshold': None,  # wird nachher gesetzt
        }
    # Ladeprofil-DataFrame bauen
    if all_charging_entries:
        df_charging = pd.DataFrame(all_charging_entries, columns=['time_', 'cid', 'power'])
        load_profile_df = df_charging.pivot_table(index='time_', columns='cid', values='power', aggfunc='sum').fillna(0)
        load_profile_df.index.name = 'time_'
        # Statistiken pro cid berechnen
        for cid in unique_cids:
            if cid in load_profile_df:
                max_load = load_profile_df[cid].max()
                above_threshold = (load_profile_df[cid] > load_threshold)
                above_threshold_per_day = above_threshold.groupby(load_profile_df.index.date).sum()
                avg_minutes_above_threshold = above_threshold_per_day.mean() if not above_threshold_per_day.empty else 0
                charging_stats[cid]['max_load_kW'] = max_load
                charging_stats[cid]['avg_minutes_above_threshold'] = round(avg_minutes_above_threshold, 1)
    else:
        load_profile_df = pd.DataFrame()
    return load_profile_df, charging_stats


#TODO: Version Georg 2025-07-25 

# --- Alte Funktion auskommentiert ---
# @cache.cache
# def calculate_charging_load_profiles(df_, charging_power, load_threshold, charging_strategy='immediate'):
#     """
#     Calculate charging load profiles for each charging station based on tour data.

#     Input: DataFrame with energy demand and stop times for each tour
#         e.g output/track_energies/tours_constant_charging_100-300_no_disp.csv
        
#     Parameters:
#     -----------
#     df_tours : DataFrame
#         DataFrame containing tour data with energy demand and stop times
#     charging_powers : dict
#         Dictionary mapping location types to charging power values
#     load_threshold : float, optional
#         Threshold in kW to calculate time above threshold, default is 630 kW
#     charging_strategy : str, optional
#         Charging strategy to use: 'immediate' (default), 'delayed', or 'average'
        
#     Returns:
#     --------
#     cid_load_profiles : dict
#         Dictionary of DataFrames, one for each charging station ID (cid),
#         containing the charging load cid stats (as in original)
#     charging_stats : dict
#         Dictionary containing statistics for each CID (avg duration, max load, time above threshold)
#     """
#     df = df_.loc[(df_.location == 'home base') & (df_.distance.isna())]
    
#     # Round stop and start times to the nearest minute
#     df = df.assign(stop_time_=df['stop_time'].dt.floor('min'))
  
#     if charging_strategy == 'delayed':
#         # Adjust start charging time to be as late as possible before stop_time
#         df = df.assign(charging_duration_min=(df['energy_recharged_kwh'] / charging_power * 60).astype(int))
#         df = df.assign(
#             start_charging_time=df['stop_time_'] + pd.to_timedelta((df['duration'] / 60 - df['charging_duration_min']).fillna(0).astype(int), unit='m')
#         )
#         df = df.assign(charging_power=charging_power)
#     elif charging_strategy == 'average':
#         # Charge evenly over the whole time frame from start_charging_time to stop_time_
#         # Here, we assume start_charging_time is the earliest time charging can start, e.g. arrival time
#         # For simplicity, assume charging window is from (stop_time_ - charging_duration_min) to stop_time_
#         # So average power is charging_power spread evenly over charging_duration_min
#         df = df.assign(charging_duration_min=(df['duration'] / 60).astype(int))
#         df = df.assign(start_charging_time=df['stop_time_'])
#         df = df.assign(charging_power=df['energy_recharged_kwh'] / df['charging_duration_min'] * 60)
#     elif charging_strategy == 'immediate':
#         # Immediate charging: start charging at stop_time_
#         df = df.assign(charging_duration_min=(df['energy_recharged_kwh'] / charging_power * 60).astype(int))
#         df = df.assign(start_charging_time=df['stop_time_'])
#         df = df.assign(charging_power=charging_power)
#     else:
#         raise ValueError("Invalid charging strategy. Choose from 'immediate', 'delayed', or 'average'.")
    
#     df = df.assign(end_charging_time=df['start_charging_time'] + pd.to_timedelta(df['charging_duration_min'], unit='m'))
    
#     # Expand df to one row per minute of charging using the charging_duration_min column
#     df_expanded = df.loc[df.index.repeat(df['charging_duration_min'])].copy()
#     df_expanded['minute_offset'] = df_expanded.groupby(level=0).cumcount()

#     df_expanded['time_'] = df_expanded['start_charging_time'] + pd.to_timedelta(df_expanded['minute_offset'], unit='m')
    
#     load_agg = df_expanded.groupby(['cid', 'time_']).charging_power.sum().unstack(level=0).fillna(0)
    
#     # Prepare output dictionaries
#     cid_load_profiles = {}
#     charging_stats = {}
   
#     # For each cid, create load profile DataFrame and calculate statistics
#     charging_stats = {
#         'max_load_kW': load_agg.max().to_dict(),
#         'minutes_above_threshold': (load_agg > load_threshold).sum().to_dict(),
#     }
        
#     return load_agg, charging_stats

