# _________________________________
# BET.OS Function #
# Operation Scenario Block #
# Designed by MX-2023-02-03
# Modified Imports by Philip Rosborough on 2025-02-10
# Function Description
# Preparation Truck Tracks from Smart Mobility Database
# Part of the Freight Module
# _________________________________

# import functions

# import libraries
import pandas
import numpy as np
from scipy.signal import savgol_filter
import sqlalchemy
from energy_sim_plots import plot_alt_profile, plot_alt_and_speed_profile

# ______________________________________________________________________________________________________________________
# Read Tracks from SQL Database
# TODO: entfernen / durch config ersetzen / gitignore
def get_tracks(scenario):
    # Create SQL Engine
    # create database connection and query data
    engine = sqlalchemy.create_engine('postgresql://{username}:{password}@{url}:{port}/{db_name}'.format(
        username='ftm_asc_paper',   #'ftm_asc_zaehringer',
        url='postgres-sm.ftm.ed.tum.de', #postgres-sm.ftm.ed.tum.de
        password='u436UDyanxDuDiMUpATu5eKb',   #'JurmMc2Jjj4uCJpckhf',
        db_name='mobtrack',
        port=5432))
    
    track = pandas.read_sql(
        f'''select _time as time, altitude, speed from public.ftm_reduced_gps_from_track_id({str(scenario.track_id[0])},1)''', 
        con=engine, 
        index_col='time'
        ) 
    
    try :
        track.iloc[0]
    except:
        print(f"Track {scenario['track_id'][0]} is empty")
        return None, None, None, None

    speed_profile = track.speed.to_numpy()
    altitude_profile = track.altitude.to_numpy()
    altitude_profile_filter = savgol_filter(altitude_profile, window_length=291, polyorder=4) #Reduced Window length from 1001 to 291
    track_duration = len(speed_profile)
    # <>
    # Calculate Alpha over Time
    # alpha = asin(h2-h1/(((v2+v1)/2)*t)
    h1= altitude_profile_filter[0:len(altitude_profile)-1]
    h2= altitude_profile_filter[1:len(altitude_profile)]
    v1= speed_profile[0:len(speed_profile)-1]
    v2= speed_profile[1:len(speed_profile)]

    # Calculate Alpha, but filter out rows where v1+v2/2 is zero (i.e the velocity at two consecutive timestamps is zero)
    # Still get runtime warnings here, because we sometimes divide through very small distances -> alpha_reduced gets very large
    # but route_array_slope still has reasonable values
    alpha_reduced = np.where((v2 + v1) != 0, ((h2 - h1) / (((v2 + v1) / 2) * 1)) * 100, 0)

    alpha_array = np.concatenate([alpha_reduced, np.array([0])])
    alpha_array =np.nan_to_num(alpha_array)
    # <>
    # Calculate Distance based Track
    # Calculate Distance over time
    i = 0
    array_distance_raw = np.zeros(track_duration)

    while i < track_duration - 1:
        # Set velocity exactly zero in the area of zer0
        if speed_profile[i] <= 1 and speed_profile[i+1] <= 1:
            speed_profile[i] = 0.0
            speed_profile[i+1] = 0.0

        array_distance_raw[i + 1] = array_distance_raw[i] + ((speed_profile[i + 1] + speed_profile[i]) / 2) * 1  # 1Hz Sample
        # Update Variable
        i = i + 1

    # set all array values to None if 50 % of the speed profile is rounded down to zero 
    # and subsequently skip the energy simulation
    # 
    if np.sum(speed_profile == 0) / len(speed_profile) >= 0.5:
        print(f"Rounded speed profile of track {scenario['track_id'][0]} is mostly Zero")
        return None, None, None, None
    #"""
    # Before plotting the altitude profiles, the track_ids have to be mapped 
    # e.g. 
    track_id_map = np.array([
        #[1500000384317, 10786],
        #[1500000376306, 23894],
        #[1500000628544, 119905],
        #[1500000628588, 119947],
        [1500000635173, 126509],
        #[1500000604095, 144124],
        #[1500000596648, 144175],
        #[1500000593056, 148066]
    ])
    #"""

    # plot_alt_profile(
    #     scenario = scenario, 
    #     track = track, 
    #     altitude_profile = altitude_profile, 
    #     altitude_profile_filter = altitude_profile_filter, 
    #     track_id_map = track_id_map)

    plot_alt_and_speed_profile(
        scenario = scenario, 
        track = track, 
        altitude_profile = altitude_profile, 
        altitude_profile_filter = altitude_profile_filter, 
        track_id_map = track_id_map,
        speed_profile = speed_profile)

    # Get Total Distance of Cycle
    distance_cycle = int(np.floor(array_distance_raw[-1]))  # in m
    # Floor array_distance raw
    array_distance_raw = np.floor(array_distance_raw)
    # Initialize distancebased Variables
    array_speed_distancebased_raw = np.zeros(distance_cycle)
    array_distance_distancebased = np.zeros(distance_cycle)
    array_stoptime_distancebaseed = np.zeros(distance_cycle)
    array_slope_distancebased = np.zeros(distance_cycle)

    # Safe Velocity in each Meter Step for distance based Variables
    dis = 0
    k = 0

    while dis < distance_cycle:
        while dis < array_distance_raw[k]:
            array_speed_distancebased_raw[dis] = speed_profile[k]
            if speed_profile[k] <= 3:
                array_slope_distancebased[dis] = 0
            else:
                array_slope_distancebased[dis] = alpha_array[k]
            array_distance_distancebased[dis] = dis
            dis = dis + 1
        k = k + 1

    # Distancebased Variable
    route_array_distance = array_distance_distancebased
    route_array_speed = array_speed_distancebased_raw
    route_array_stoptime = array_stoptime_distancebaseed
    route_array_slope = array_slope_distancebased
    
    # Smooth Speed Profile
    # route_array_speed would be empty (e.g of length 0), when the entire speed profile is rounded down to zero
    # thus we skip the energy simulation if this is the case (see above) 
    route_array_speed = savgol_filter(route_array_speed, window_length=101, polyorder=3)
    route_array_speed = np.maximum(route_array_speed, 0)  # No negative Speed due to Filter function
    # Check Speed Profile
    route_array_speed[(route_array_speed == 0)] = 1/3.6
     
    return route_array_speed, route_array_distance, route_array_stoptime, route_array_slope
