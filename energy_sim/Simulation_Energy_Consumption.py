# Skript not included in Betos but provided by Maximilian Zähringer (Author of Betos)
# Modified by Philip Rosborough to include handling of zero speed profiles

from BETOS_Framework.M1_Vehicle_Properties import M1_Veh_Prop
from BETOS_Framework.M2_Environment_Properties import M2_Env_Prop
from BETOS_Framework.M3_Freight_Properties import M3_Freight_Prop
from BETOS_Framework.M4_Static_Vehicle import M4_Veh_Static
from BETOS_Framework.M5_Static_Environment import M5_Env_Static
from BETOS_Framework.M9_Operation_Strategy.M9_1_Prediction import M91_Prediction

import numpy as np

# Main Function
def sim_energy_con(scenario):
    # Preprocessing: Load Freight Mission Data, Get Vehicle up for Simulation
    #print('Preprocessing started')
    # Do some shallow work for running object based code
    vehicle, env, sim = object_def()
    # Module M1: Vehicle Properties
    vehicle = M1_Veh_Prop.vehicle_properties(vehicle, scenario)
    # Module M2: Environment Properties
    env = M2_Env_Prop.environment_properties(env, scenario)
    # Module M3: Freight Properties
    sim, env = M3_Freight_Prop.freight_properties(sim, env, scenario)

    # If the route only has speed values of 0, freight_properties() will assign None to the arrays and the 
    # energy simulation should be skipped
    if (env.route_array_distance is None and 
        env.route_array_speed is None and 
        env.route_array_stoptime is None and 
        env.route_array_slope is None):
        sim.energy_consumption_kwhkm = None 
        sim.energy_consumption_kwh = None
        print('Route has no speed values or is empty, skipping energy simulation [Simulation_Energy_Consumption.py]')
        return sim

    # Module M4: Building Vehicle
    sim.betos_manipulation_charge = 0
    vehicle = M4_Veh_Static.static_vehicle_model(sim, vehicle, env)
    # Module M5: Building Environment
    env = M5_Env_Static.static_env(env, scenario)
    # Initialization of Simulation Variables
    sim = initialization(env, sim, scenario)

    # Set Manipualtion Parameter
    sim.betos_manipulation_cons = 0
    sim.betos_manipulation_time = 0
    sim.betos_manipulation_charge = 0

    sim = M91_Prediction.betos_prediction_energy_consumption(sim, vehicle, env)

    sim.energy_consumption_kwhkm = sum(sim.betos_energy_cons_dis_prediction)/sim.dis_steps/3600
    sim.energy_consumption_kwh = sum(sim.betos_energy_cons_dis_prediction) / 3600000

    return sim


# Getting Object Based Code running:
def object_def():
    # Truck Class and Vehicle Object
    class Truck:
        pass
    vehicle = Truck()

    # Environment Class and env object
    class Environment:
        pass
    env = Environment()

    # Simulation Class and sim object
    class Simulation:
        pass
    sim = Simulation()

    return vehicle, env, sim


# Do some Initialization Work for Operation Simulation
def initialization(env, sim, scenario):
    # Simulation steps in distance
    sim.dis_steps = int(len(env.route_array_distance))
    # Simulation steps in time are unknow due to charging events, assuming double dis steps
    sim.time_steps = sim.dis_steps * 2
    # Run Variable in distance
    sim.step_dis = 0  # starting at 0 meter
    # Run Variable in time
    sim.step_time = 0  # starting at 0 seconds
    # Daytime while simulation
    sim.daytime = env.freight_mission_start  #
    # Critical PoI for DP Approach
    sim.pos_poi_crit = 0
    # Position of done rest due to charging or resting
    sim.pos_rest_done = 0
    # Position of overnight stay
    sim.pos_overnight_stay = 0
    # Position of maximum reachable POI along Route
    sim.pos_poi_max = 3000000  # set initial to 3000 km
    # Aging Evaluation yes or no
    sim.aging_eval = scenario['aging'][0]

    # Driving Variables
    # Version
    sim.driving_version = scenario['driving'][0]
    # Acceleration Target in distance
    sim.acc_target_dis = np.zeros(sim.dis_steps)
    # Acceleration in distance
    sim.acc_dis = np.zeros(sim.dis_steps)
    # Acceleration in time
    sim.acc_time = np.zeros(sim.time_steps)
    # Velocity in distance
    sim.vel_dis = np.zeros(sim.dis_steps)
    # Velocity in time
    sim.vel_time = np.zeros(sim.time_steps)
    # Time Vector distancebased
    sim.time_dis = np.zeros(sim.dis_steps)
    # Time Vector timebased
    sim.time_time = np.zeros(sim.time_steps)
    # Charging time distancebased
    sim.charge_time_dis = np.zeros(sim.dis_steps)
    # Rest Time distancebased
    sim.rest_time_dis = np.zeros(sim.dis_steps)
    # Waiting Time distancebased
    sim.wait_time_dis = np.zeros(sim.dis_steps)

    # BETOS Variables
    # Action Vector
    sim.betos_action = np.zeros(sim.dis_steps)
    # SOC Target Vector
    sim.betos_soc_target = np.zeros(sim.dis_steps)
    # Target Charging Time
    sim.betos_charge_time = np.zeros(sim.dis_steps)
    # Rest Time Vector
    sim.betos_rest_time = np.zeros(sim.dis_steps)
    # Track Deep Discharging Events
    sim.betos_deep_discharge = 0
    # Track Stop Events in Total
    sim.betos_stops = 0
    # Run empty using Strategy
    sim.betos_run_empty = 0
    # Re Parking allowed while waiting of occupied charging station (0 No, 1 Yes)
    sim.betos_re_park = scenario['re_park'][0]
    # DP run variable
    sim.betos_dp_run = 0
    # Wait count
    sim.betos_wait_count = 0
    # Availability of chosen POI
    sim.betos_poi_availability = np.zeros(sim.dis_steps)
    # Track charging power of choosen PoI
    sim.betos_power = np.zeros(sim.dis_steps)
    # Recharged Energy at POI in kWh
    sim.betos_charge_amount = np.zeros(int(env.infra_number_poi))
    # Arrival time at POI
    sim.betos_poi_arrival_time = np.zeros(int(env.infra_number_poi))
    # Total rest time at POI in h
    sim.betos_down_time = np.zeros(int(env.infra_number_poi))
    # Version of BETOS
    sim.betos_version = scenario['version'][0]
    # Predicted driving time to destination from POI i
    sim.betos_driving_time_destination = np.zeros((int(env.infra_number_poi), 2))

    # Battery Variables
    # Battery Power in distance
    sim.bat_power_dis = np.zeros(sim.dis_steps)
    # Battery Power in time
    sim.bat_power_time = np.zeros(sim.time_steps)
    # Battery Temperature in distance
    sim.bat_temp_dis = np.zeros(sim.dis_steps)
    # Battery Temperature in time
    sim.bat_temp_time = np.zeros(sim.time_steps)
    # Battery SoC in distance
    sim.bat_soc_dis = np.zeros(sim.dis_steps)
    sim.bat_soc_dis[0] = env.freight_start_soc
    # Battery SoC in time
    sim.bat_soc_time = np.zeros(sim.time_steps)
    sim.bat_soc_time[0] = env.freight_start_soc
    # Battery SoH in distance
    sim.bat_soh_dis = np.zeros(sim.dis_steps)
    # Battery SoH in time
    sim.bat_soh_time = np.zeros(sim.time_steps)
    # Energy Consumption in distance
    sim.energy_cons_dis = np.zeros(sim.dis_steps)
    # Aging evaluation 0:No, 1:Yes
    sim.aging = scenario['aging'][0]

    return sim









