#  Truck Fleet Electrification Analysis

A comprehensive research framework for analyzing the electrification potential of commercial truck fleets, including energy consumption simulation, charging infrastructure requirements, and operational optimization strategies.

## Created by
Anna Paper, M.Sc.  
Institute of Automotive Technology  
Department of Mobility Systems Engineering  
TUM School of Engineering and Design  
Technical University of Munich  
anna.paper@tum.de  
2025

#### Contributors
Georg Balke, M.Sc. - Research Associate 07/2021 - 12/2025

Philip Rosborough, B.Sc. - Semester thesis 11/2024 - 06/2025

## Overview

This repository contains the complete analysis framework for studying truck fleet electrification, including:

- **Data Processing**: GPS tracking data analysis and tour reconstruction (CSV-based)
- **Energy Simulation**: Battery electric truck energy consumption modeling using BETOS framework
- **Charging Analysis**: Load profile generation and charging infrastructure requirements
- **Operational Optimization**: Fleet operation strategies and charging management
- **Visualization**: Professional plotting and analysis tools

**Important**: This framework has been updated to use CSV files instead of database queries, making it more accessible and easier to deploy without database setup requirements.

## Installation

#### Step 1: Getting the source code
Clone the repository from GitHub:
```bash
git clone https://github.com/[your-username]/sm-truck-fleet-electrification-paper.git # TODO: Anpassen
cd sm-truck-fleet-electrification-paper # TODO: Anpassen
```

#### Step 2: Create a clean virtual environment
It is recommended to create and activate a clean virtual environment for the installation. This can be done using conda:
```bash
conda create -n truck_electrification python=3.8 # TODO: Anpassen
conda activate truck_electrification # TODO: Anpassen

Or alternatively with venv:
```bash
python -m venv truck_electrification_env # TODO: Anpassen 
source truck_electrification_env/bin/activate  # On Windows: truck_electrification_env\Scripts\activate # TODO: Anpassen
```

#### Step 3: Install dependencies
Install all required packages using the provided environment file:
```bash
conda env update -f environment.yml
```

Or install manually (CSV-based workflow):
```bash
pip install pandas==2.0.3 seaborn matplotlib==3.7.2 plotly ipykernel ipywidgets scienceplots
```

**Note**: The following packages are no longer required for the CSV-based workflow:
- sqlalchemy, geopandas, psycopg2 (database operations)
- shapely, contextily (spatial analysis)
- fpdf, svgpathtools, svgpath2mpl (legacy utilities)

## Project Structure

```
sm-truck-fleet-electrification-paper/
├── data_handling/              # Data processing and acquisition modules
│   ├── data_processing.py      # Core data processing functions
│   ├── data_recordings.py      # Data recording utilities
│   └── data-aquisition.ipynb   # Data acquisition notebook
├── energy_sim/                 # Energy simulation framework
│   ├── BETOS_Framework/        # BETOS energy simulation framework
│   ├── sequential_analysis.py  # Sequential analysis functions
│   ├── load_profile.py         # Charging load profile generation
│   └── energy_sim_functions.py # Energy simulation utilities
├── utils/                      # Utility functions and configuration
│   ├── style_config.py         # Plotting style configuration
│   └── utilities.py            # Database utilities (OBSOLETE - CSV-based workflow)
├── input/                      # Input data files
│   ├── home/                   # Fleet data
│   └── stations/               # Charging station data
├── output/                     # Generated outputs
│   ├── figures/                # Generated plots and visualizations
│   ├── csvs/                   # Processed data files
│   └── charging_loads/         # Charging load profiles
├── main_data_analysis.ipynb    # Main data analysis notebook
├── main_energy_sim.ipynb       # Main energy simulation notebook
└── environment.yml             # Conda environment specification
```

## Usage

### Basic Workflow

The analysis follows a sequential workflow:

1. **Data Acquisition** (`data_handling/data-aquisition.ipynb`)
   - Load data from CSV files (no database required)
   - Process GPS tracking data
   - Extract tour and track information
   - Clean and validate data

2. **Data Analysis** (`main_data_analysis.ipynb`)
   - Analyze fleet operational patterns
   - Generate descriptive statistics
   - Create operational visualizations

3. **Energy Simulation** (`main_energy_sim.ipynb`)
   - Simulate energy consumption for each track
   - Calculate battery state of charge (SoC) profiles
   - Generate charging load profiles
   - Analyze charging infrastructure requirements

### Running the Analysis

#### Option 1: Jupyter Notebooks (Recommended)
```bash
jupyter notebook
```
Then open the notebooks in the following order:
1. `data_handling/data-aquisition.ipynb`
2. `main_data_analysis.ipynb`
3. `main_energy_sim.ipynb`

#### Option 2: Command Line
```bash
python -m jupyter notebook main_data_analysis.ipynb
python -m jupyter notebook main_energy_sim.ipynb
```

### Key Features

- **Multi-Scenario Analysis**: Compare different charging strategies and battery configurations
- **Load Profile Generation**: Calculate charging demand profiles for infrastructure planning
- **Fleet Optimization**: Analyze operational patterns and optimization opportunities
- **Professional Visualization**: Publication-ready plots with TUM styling

## Configuration

### Scenario Parameters
The analysis supports multiple scenarios defined in `main_energy_sim.ipynb`:

- **Default**: 572 kWh battery, 150 kW home charging
- **Big Battery**: 798 kWh battery, 150 kW home charging
- **Destination Charging**: 572 kWh battery, 150 kW home + 350 kW destination charging
- **Low Home**: 572 kWh battery, 50 kW home charging
- **High Home**: 572 kWh battery, 350 kW home charging
- **Ultra Home**: 572 kWh battery, 10 MW home charging

### Style Configuration
Professional plotting styles are configured in `utils/style_config.py` with TUM color schemes and publication-ready formatting.

## Output Files

The analysis generates several output files:

- **Figures**: PDF and SVG plots in `output/figures/`
- **Data**: Processed CSV files in `output/csvs/`
- **Load Profiles**: Charging demand profiles in `output/charging_loads/`
- **Tour Data**: Aggregated tour statistics and energy consumption

## Dependencies

### Core Dependencies
- Python 3.8.12
- pandas 2.0.3
- matplotlib 3.7.2
- seaborn
- numpy

### Specialized Dependencies
- BETOS Framework (included in energy_sim/)
- Plotly (for interactive visualizations)

### Legacy Dependencies (OBSOLETE - CSV-based workflow)
- SQLAlchemy (was used for database operations - no longer required)
- GeoPandas (was used for spatial analysis - no longer required)
- psycopg2 (was used for PostgreSQL connection - no longer required)

## Contributing and Support

For contributing to the code please contact:

Anna Paper Institute of Automotive Technology
Technical University of Munich

mail: anna.paper@tum.de

## Versioning

V1.1


## License and Copyright

Copyright 2025 Anna Paper

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Citation

If you use this code in your research, please cite the corresponding article and dataset:

[```bibtex
@article{Paper2025, 
  title={Assessing battery electric truck feasibility in heavy-duty logistics fleets: Insights from GPS-based energy simulations and charging load profiles},
  author={Paper, Anna and Balke, Georg and Rosborough, Philip and Brödel, Marcel and Klein, Niclas and Lienkamp, Markus},
  journal={Journal Name}, #TODO: ergänzen
  year={2025}, #TODO: ergänzen
  publisher={Publisher} #TODO: ergänzen
}
```]

Dataset: https://zenodo.org/records/16411298

## Acknowledgments

- Technical University of Munich for research support
- BETOS Framework developers
- Contributors and research partners 
