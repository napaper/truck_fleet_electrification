"""
Energy consumption simulation for truck fleet electrification analysis.

This module provides the main energy consumption simulation functionality using the
BETOS framework. It integrates vehicle properties, environment conditions, and
freight mission data to calculate energy consumption for electric truck operations.

The module was originally provided by Maximilian Zähringer (Author of BETOS) and
has been modified by Philip Rosborough to include handling of zero speed profiles.

Key Features:
- Integration with BETOS framework modules
- Vehicle and environment property initialization
- Energy consumption calculation for driving missions
- Support for zero-speed route handling
- Comprehensive simulation variable initialization
"""

from BETOS_Framework.M1_Vehicle_Properties import M1_Veh_Prop
from BETOS_Framework.M2_Environment_Properties import M2_Env_Prop
from BETOS_Framework.M3_Freight_Properties import M3_Freight_Prop
from BETOS_Framework.M4_Static_Vehicle import M4_Veh_Static
from BETOS_Framework.M5_Static_Environment import M5_Env_Static
from BETOS_Framework.M9_Operation_Strategy.M9_1_Prediction import M91_Prediction

import numpy as np


# =============================================================================
# MAIN ENERGY CONSUMPTION SIMULATION
# =============================================================================

def sim_energy_con(scenario):
    """
    Execute the main energy consumption simulation for a given scenario.
    
    This function orchestrates the complete energy consumption simulation by
    initializing vehicle and environment objects, processing freight mission data,
    and executing the BETOS prediction algorithm to calculate energy consumption.
    
    Args:
        scenario (dict): Scenario configuration containing vehicle, environment,
            and operation parameters for the simulation
            
    Returns:
        Simulation: Simulation object with calculated energy consumption values:
            - energy_consumption_kwhkm: Energy consumption per kilometer
            - energy_consumption_kwh: Total energy consumption for the mission
            
    Note:
        The function automatically skips simulation for routes with zero speed
        values or empty route data, returning None values for energy consumption.
        This prevents errors when processing invalid or stationary routes.
    """
    # Preprocessing: Load Freight Mission Data, Get Vehicle up for Simulation
    # print('Preprocessing started')
    
    # Initialize object-based code structure
    vehicle, env, sim = object_def()
    
    # Module M1: Vehicle Properties
    # Initialize vehicle characteristics based on scenario configuration
    vehicle = M1_Veh_Prop.vehicle_properties(vehicle, scenario)
    
    # Module M2: Environment Properties
    # Set up environmental conditions and constraints
    env = M2_Env_Prop.environment_properties(env, scenario)
    
    # Module M3: Freight Properties
    # Process freight mission data and route information
    sim, env = M3_Freight_Prop.freight_properties(sim, env, scenario)

    # Check for zero-speed routes and skip simulation if necessary
    # If the route only has speed values of 0, freight_properties() will assign None
    # to the arrays and the energy simulation should be skipped
    if (env.route_array_distance is None and 
        env.route_array_speed is None and 
        env.route_array_stoptime is None and 
        env.route_array_slope is None):
        
        # Set energy consumption to None for invalid routes
        sim.energy_consumption_kwhkm = None 
        sim.energy_consumption_kwh = None
        print('Route has no speed values or is empty, skipping energy simulation [Simulation_Energy_Consumption.py]')
        return sim

    # Module M4: Building Vehicle
    # Initialize static vehicle model with charging manipulation parameters
    sim.betos_manipulation_charge = 0
    vehicle = M4_Veh_Static.static_vehicle_model(sim, vehicle, env)
    
    # Module M5: Building Environment
    # Set up static environment configuration
    env = M5_Env_Static.static_env(env, scenario)
    
    # Initialize simulation variables and parameters
    sim = initialization(env, sim, scenario)

    # Set manipulation parameters for BETOS simulation
    sim.betos_manipulation_cons = 0
    sim.betos_manipulation_time = 0
    sim.betos_manipulation_charge = 0

    # Execute BETOS energy consumption prediction algorithm
    sim = M91_Prediction.betos_prediction_energy_consumption(sim, vehicle, env)

    # Calculate final energy consumption metrics
    # Energy consumption per kilometer (kWh/km)
    sim.energy_consumption_kwhkm = sum(sim.betos_energy_cons_dis_prediction) / sim.dis_steps / 3600
    
    # Total energy consumption for the mission (kWh)
    sim.energy_consumption_kwh = sum(sim.betos_energy_cons_dis_prediction) / 3600000

    return sim


# =============================================================================
# OBJECT INITIALIZATION AND CLASS DEFINITIONS
# =============================================================================

def object_def():
    """
    Initialize the object-based code structure for BETOS simulation.
    
    This function creates the basic class objects (Truck, Environment, Simulation)
    that are required for the BETOS framework to function properly. These objects
    serve as containers for various simulation parameters and state variables.
    
    Returns:
        tuple: Contains three initialized objects:
            - vehicle: Truck object for vehicle properties
            - env: Environment object for environmental conditions
            - sim: Simulation object for simulation state and results
    """
    # Truck Class and Vehicle Object
    class Truck:
        """Container class for vehicle properties and characteristics."""
        pass
    vehicle = Truck()

    # Environment Class and env object
    class Environment:
        """Container class for environmental conditions and constraints."""
        pass
    env = Environment()

    # Simulation Class and sim object
    class Simulation:
        """Container class for simulation state, variables, and results."""
        pass
    sim = Simulation()

    return vehicle, env, sim


# =============================================================================
# SIMULATION INITIALIZATION AND VARIABLE SETUP
# =============================================================================

def initialization(env, sim, scenario):
    """
    Initialize simulation variables and parameters for energy consumption calculation.
    
    This function sets up all necessary simulation variables including distance
    and time steps, driving parameters, BETOS-specific variables, and battery
    state variables. It prepares the simulation environment for the energy
    consumption calculation process.
    
    Args:
        env (Environment): Environment object with route and infrastructure data
        sim (Simulation): Simulation object to be initialized
        scenario (dict): Scenario configuration with operation parameters
            
    Returns:
        Simulation: Fully initialized simulation object with all required variables
            and parameters set for energy consumption simulation
    """
    # Simulation steps and dimensions
    # Number of simulation steps based on route distance
    sim.dis_steps = int(len(env.route_array_distance))
    
    # Simulation steps in time (unknown due to charging events, assuming double distance steps)
    sim.time_steps = sim.dis_steps * 2
    
    # Run variables for distance-based simulation
    sim.step_dis = 0  # Starting at 0 meters
    sim.step_time = 0  # Starting at 0 seconds
    
    # Daytime during simulation
    sim.daytime = env.freight_mission_start
    
    # Critical Points of Interest (POI) for Dynamic Programming approach
    sim.pos_poi_crit = 0
    
    # Position tracking for completed rest periods (charging or resting)
    sim.pos_rest_done = 0
    
    # Position of overnight stay locations
    sim.pos_overnight_stay = 0
    
    # Position of maximum reachable POI along the route
    sim.pos_poi_max = 3000000  # Set initial to 3000 km
    
    # Aging evaluation flag (0: No, 1: Yes)
    sim.aging_eval = scenario['aging'][0]

    # Driving Variables and Parameters
    # Driving strategy version
    sim.driving_version = scenario['driving'][0]
    
    # Acceleration profiles and targets
    sim.acc_target_dis = np.zeros(sim.dis_steps)  # Target acceleration in distance
    sim.acc_dis = np.zeros(sim.dis_steps)         # Actual acceleration in distance
    sim.acc_time = np.zeros(sim.time_steps)      # Acceleration in time
    
    # Velocity profiles
    sim.vel_dis = np.zeros(sim.dis_steps)        # Velocity in distance
    sim.vel_time = np.zeros(sim.time_steps)      # Velocity in time
    
    # Time vectors for different simulation bases
    sim.time_dis = np.zeros(sim.dis_steps)       # Time vector (distance-based)
    sim.time_time = np.zeros(sim.time_steps)     # Time vector (time-based)
    
    # Activity time tracking
    sim.charge_time_dis = np.zeros(sim.dis_steps)  # Charging time (distance-based)
    sim.rest_time_dis = np.zeros(sim.dis_steps)    # Rest time (distance-based)
    sim.wait_time_dis = np.zeros(sim.dis_steps)    # Waiting time (distance-based)

    # BETOS Framework Variables
    # Action and control vectors
    sim.betos_action = np.zeros(sim.dis_steps)                    # Action vector
    sim.betos_soc_target = np.zeros(sim.dis_steps)               # SOC target vector
    sim.betos_charge_time = np.zeros(sim.dis_steps)              # Target charging time
    sim.betos_rest_time = np.zeros(sim.dis_steps)                # Rest time vector
    
    # Event tracking and counters
    sim.betos_deep_discharge = 0                                 # Track deep discharging events
    sim.betos_stops = 0                                          # Track total stop events
    sim.betos_run_empty = 0                                      # Run empty using strategy
    sim.betos_wait_count = 0                                     # Wait count
    
    # Charging station and infrastructure parameters
    sim.betos_re_park = scenario['re_park'][0]                   # Re-parking allowed (0: No, 1: Yes)
    sim.betos_dp_run = 0                                         # Dynamic Programming run variable
    sim.betos_poi_availability = np.zeros(sim.dis_steps)         # Availability of chosen POI
    sim.betos_power = np.zeros(sim.dis_steps)                    # Track charging power of chosen POI
    
    # POI-specific tracking arrays
    sim.betos_charge_amount = np.zeros(int(env.infra_number_poi))           # Recharged energy at POI (kWh)
    sim.betos_poi_arrival_time = np.zeros(int(env.infra_number_poi))       # Arrival time at POI
    sim.betos_down_time = np.zeros(int(env.infra_number_poi))              # Total rest time at POI (hours)
    
    # Version and prediction parameters
    sim.betos_version = scenario['version'][0]                   # Version of BETOS
    sim.betos_driving_time_destination = np.zeros((int(env.infra_number_poi), 2))  # Predicted driving time to destination from POI

    # Battery State Variables
    # Power profiles
    sim.bat_power_dis = np.zeros(sim.dis_steps)                  # Battery power (distance-based)
    sim.bat_power_time = np.zeros(sim.time_steps)                # Battery power (time-based)
    
    # Temperature profiles
    sim.bat_temp_dis = np.zeros(sim.dis_steps)                   # Battery temperature (distance-based)
    sim.bat_temp_time = np.zeros(sim.time_steps)                 # Battery temperature (time-based)
    
    # State of Charge (SoC) profiles
    sim.bat_soc_dis = np.zeros(sim.dis_steps)                    # Battery SoC (distance-based)
    sim.bat_soc_dis[0] = env.freight_start_soc                   # Initialize with starting SoC
    sim.bat_soc_time = np.zeros(sim.time_steps)                  # Battery SoC (time-based)
    sim.bat_soc_time[0] = env.freight_start_soc                  # Initialize with starting SoC
    
    # State of Health (SoH) profiles
    sim.bat_soh_dis = np.zeros(sim.dis_steps)                    # Battery SoH (distance-based)
    sim.bat_soh_time = np.zeros(sim.time_steps)                  # Battery SoH (time-based)
    
    # Energy consumption tracking
    sim.energy_cons_dis = np.zeros(sim.dis_steps)                # Energy consumption (distance-based)
    
    # Aging evaluation flag (0: No, 1: Yes)
    sim.aging = scenario['aging'][0]

    return sim









