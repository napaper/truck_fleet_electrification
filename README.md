# Truck Fleet Electrification Analysis

**Assessing battery electric truck feasibility in heavy-duty logistics fleets: Insights from GPS-based energy simulations and charging load profiles**

A comprehensive open-source framework for analyzing the electrification potential of commercial truck fleets using real-world 10 Hz GPS data from six German logistics companies (2.15 million km, 44,386 operating hours, 150 N3-class trucks).

## Recent Updates (May 2026)
This repository has been updated for the **resubmission to the journal *eTransportation***:

- **Energy Shares Analysis** (new): Calculation of actual *energy* distribution (home base / destination / public charging) across all six scenarios (instead of only charging events).
- **Sensitivity Analysis** (new): Full sensitivity study on payload (±20 %) and auxiliary load (±2 kW). Results saved as `sensitivity_results.csv`.

## Overview
The framework includes:
- GPS data processing and tour reconstruction
- Battery electric truck energy consumption simulation (BETOS framework)
- State-of-Charge (SoC) dynamics and intelligent charging strategies
- Minute-resolution charging load profile generation
- Sensitivity and scenario analysis

**Key Features**
- Fully CSV-based workflow (no database required)
- Publication-ready visualizations with TUM styling
- Open-source and fully reproducible

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/TUMFTM/truck_fleet_electrification.git
cd truck_fleet_electrification
```

### 2. Create and activate a virtual environment (recommended)
```bash
conda create -n truck_electrification python=3.8
conda activate truck_electrification
```

### 3. Install dependencies
```bash
pip install pandas==2.0.3 seaborn matplotlib==3.7.2 plotly ipykernel ipywidgets scienceplots numpy
```

## Project Structure
```
truck_fleet_electrification/
├── data_handling/                  # Data processing and tour reconstruction
├── energy_sim/                     # Core energy simulation
│   ├── BETOS_Framework/            # BETOS energy simulation engine
│   ├── sequential_analysis.py      # SoC calculation and charging logic
│   ├── load_profile.py             # Charging load profile generation
│   └── main_energy_sim.ipynb       # Main simulation notebook (incl. new Energy Shares + Sensitivity)
├── input/                          # Input data (CSV files)
├── output/                         # Generated results
│   ├── figures/                    # Publication-ready plots
│   ├── csvs/                       # Processed data
│   └── sensitivity_results.csv     # Sensitivity analysis output
├── main_data_analysis.ipynb        # Operational pattern analysis
├── main_energy_sim.ipynb           # Main energy + charging analysis
└── environment.yml                 # Optional conda environment
```

## Usage

### Recommended Workflow
1. `main_data_analysis.ipynb` – Fleet operational patterns and data exploration  
2. `main_energy_sim.ipynb` – Energy simulation, SoC calculation, charging load profiles, energy shares and sensitivity analysis

Open the notebooks:
```bash
jupyter notebook
```

## Citation

If you use this repository, the code, or the dataset in your research, please cite:

### Paper
```bibtex
@unpublished{Paper2026,
  title = {Assessing battery electric truck feasibility in heavy-duty logistics fleets: Insights from GPS-based energy simulations and charging load profiles},
  author = {Paper, Anna and Balke, Georg and Rosborough, Philip and Br{\"o}del, Marcel and Klein, Niclas and Winkler, Tom and Lienkamp, Markus},
  note = {Manuscript under review at eTransportation},
  year = {2026}
}
```

### Dataset
```bibtex
@dataset{trucks_anonymized_driving_2025,
  title        = {Dataset of Trucks' Anonymized Recorded Driving and Operation},
  author       = {{TUM - Institute of Automotive Technology} and {Technical University of Munich}},
  year         = {2025},
  month        = jul,
  day          = {24},
  version      = {v2},
  doi          = {10.5281/zenodo.16411298},
  url          = {https://zenodo.org/records/16411298},
  publisher    = {Zenodo}
}
```

**Zenodo Dataset:** [https://zenodo.org/records/16411298](https://zenodo.org/records/16411298)

## License
Copyright © 2025–2026 Anna Paper and contributors.  
Licensed under the Apache License, Version 2.0.


