import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt

import data_processing as dp

# Define LOCATIONS
LOCATIONS = dp.LOCATIONS

# Load data
df_trips_unfiltered, df_fleet, zf = dp.load_data()

# # Load speed data for a specific vehicle and trip
# i_veh, i_trip = 1, 10
# df_speed = dp.load_speed_data(zf, i_veh, i_trip)

# Preprocess data
df_trips = dp.preprocess_trips_data(df_trips_unfiltered, df_fleet)
df_stops = dp.process_stops_data(df_trips)
# df_speed = dp.convert_speed_to_kmh(df_speed)
df_occupation = dp.calculate_occupation(df_trips)

df_trips['start_time'] = pd.to_datetime(df_trips['start_time'], utc=True)
df_trips['stop_time'] = pd.to_datetime(df_trips['stop_time'], utc=True)

number_of_trucks = len(df_trips['vehicle_id'].unique())

df_occupation_processed = df_occupation.sort_index().copy()
df_occupation_processed['stop_time'] = df_occupation_processed['stop_time'].round('min')
df_occupation_processed.index = df_occupation_processed.index.round('min')

stop_min = df_occupation_processed.iloc[0]['stop_time']
stop_max = df_occupation_processed.iloc[-1]['stop_time']
index_min = df_occupation_processed.index.min()
index_max = df_occupation_processed.index.max()

""" more understandable but doesn't seem to work
index_min = df_occupation.index.min()
stop_min = df_occupation['stop_time'].min()
index_max = df_occupation.index.max()
stop_max = df_occupation['stop_time'].max()
"""

start_time_rec = min(stop_min, index_min).tz_convert('Europe/Berlin')
end_time_rec = max(stop_max, index_max).tz_convert('Europe/Berlin')
assert start_time_rec <= df_occupation_processed['stop_time'].min(), "start_time_rec is not less than the minimum stop time"
assert end_time_rec >= df_occupation_processed['stop_time'].max(), "end_time_rec is not greater than the maximum stop time"

# Generate a range of timestamps at a minute frequency
time_range = pd.date_range(start=start_time_rec, end=end_time_rec, freq='min', tz='Europe/Berlin')
# Create a DataFrame with all minutes
df_minutes = pd.DataFrame(time_range, columns=['minute'])
df_merged = df_minutes.merge(df_occupation_processed, left_on='minute', right_index=True, how='left')

# Forward fill the occupation data to align with each minute
df_merged['vehicle_id'] = df_merged['vehicle_id'].ffill()
df_merged['stop_time'] = df_merged['stop_time'].ffill()
df_merged['occupation'] = df_merged['occupation'].ffill()

# TODO Not the same, discuss with Anna if I can change df_occupation to have consisitent start and stop times for driving and 'stops' (e.g. home_base)
first_timestamps = df_merged.groupby('vehicle_id')['minute'].first().reset_index()
first_timestamps_2 = df_trips.groupby('vehicle_id')['start_time'].first().reset_index()

start_counting_at = first_timestamps['minute'].max()


"""
# Plot trips on timeline to see how much overlap there is between trips with each vehicle id having one row of the plot
fig, ax = plt.subplots(figsize=(15, 10))

for vehicle_id in df_trips['vehicle_id'].unique():
    vehicle_trips = df_trips[df_trips['vehicle_id'] == vehicle_id]
    for _, trip in vehicle_trips.iterrows():
        ax.plot([trip['start_time'], trip['stop_time']], [vehicle_id, vehicle_id], marker='o')

ax.set_xlabel('Time')
ax.set_ylabel('Vehicle ID')
ax.set_title('Trips Timeline')
plt.show()
"""

# Plot occupation data vs trips for one vehicle to see if they complete each other (gap of trip should be non-driving occupation)
vehicle_id = df_trips['vehicle_id'].unique()[0]  # Select the first vehicle for demonstration
vehicle_trips = df_trips[df_trips['vehicle_id'] == vehicle_id]
vehicle1_occupation = df_occupation_processed[df_occupation_processed['vehicle_id'] == vehicle_id]

fig, ax = plt.subplots(figsize=(15, 5))

for _, trip in vehicle_trips.iterrows():
    ax.plot([trip['start_time'], trip['stop_time']], [1, 1], marker='o', label='Trip')

for _, occ in vehicle1_occupation.iterrows():
    ax.plot([occ.name, occ['stop_time']], [0, 0], marker='x', label='Occupation')
    # Filter data to show only a 1 month period
    start_date = start_time_rec
    end_date = start_date + pd.DateOffset(months=1)

    # Filter trips and occupation data for the selected vehicle within the 1 month period
    vehicle_trips = vehicle_trips[(vehicle_trips['start_time'] >= start_date) & (vehicle_trips['start_time'] <= end_date)]
    vehicle1_occupation = vehicle1_occupation[(vehicle1_occupation.index >= start_date) & (vehicle1_occupation.index <= end_date)]
ax.set_xlabel('Time')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Occupation', 'Trip'])
ax.set_title(f'Occupation vs Trips for Vehicle {vehicle_id}')
plt.legend()
plt.show()

breakpoint()

# Filter for 'home_base' occupation and count vehicles at base for each minute
df_merged['at_base'] = (df_merged['minute'] <= df_merged['stop_time']) & (df_merged['occupation'] == 'home_base')
df_merged['at_base'] = df_merged['at_base'].astype(int)

# Group by minute and sum the at_base column to get the count of vehicles at base
result = df_merged.groupby('minute')['at_base'].sum()

# Delete all rows in df_merged before minute start_counting_at
df_merged = df_merged[df_merged['minute'] >= start_counting_at]
# Filter for 'home_base' occupation and count vehicles at base for each minute
df_merged['at_base'] = (df_merged['minute'] <= df_merged['stop_time']) & (df_merged['occupation'] == 'home_base')
df_merged['at_base'] = df_merged['at_base'].astype(int)

# Group by minute and sum the at_base column to get the count of vehicles at base
result_subset = df_merged.groupby('minute')['at_base'].sum()



# Options:
# (1) start with the first minute in which all trucks in df_tracks have their first entry
# (2) start with very first minute in the dataset (= start_time_rec) 
#     and maybe also plot the number of trucks that have a recording so far ('as max possible number')
#     maybe even plot relavitive (i.e percentage of trucks at homebase)
# (3) start on 



"""
Iterating takes too long for the whole dataset.
# Iterate over each minute in the timeframe
for minute in time_range: 
    at_base = 0
    print(minute)
    for index, row in df_occupation.iterrows(): 
    # index = 'start_time'; row[0] = vehicle_id, row[1] = stop_time, row[2] = occupation, row[3] = duration
        if index <= minute <= row[1] and row[2] == 'home_base':
            print('at base')
            at_base += 1
"""


    # Perform your desired operations here

"""
while minute 
for vehicle in vehicl_id:



"""