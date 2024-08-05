# HRS Location Optimization

## Overview
This project focuses on optimizing the placement of Hydrogen Refueling Stations (HRS) in the North Netherlands region. By integrating traffic intensity data with existing gas station locations, the model aims to maximize network efficiency and coverage for hydrogen fuel cell electric vehicles (FCEVs). The project's goal is to support the transition to greener transportation by providing a framework that guides policymakers and stakeholders in strategic infrastructure development.

## Folder Structure
- **`group2-model.py`**: The primary Python script containing the optimization model, data preprocessing, sensitivity analysis, and results generation.
- **`data/`**: Directory containing traffic intensity data from sensors and gas station locations.
- **`results/`**: Generated maps, heatmaps, and graphs from the sensitivity analysis, showing optimal station placements and coverage.
- **`report/`**: Contains the final report detailing the methodology, results, and conclusions of the study.

## Datasets
The project utilizes two primary datasets:
- **Traffic Intensity Data**: Retrieved from the Dutch National Traffic Data Portal (NDW), capturing road usage patterns across the North Netherlands.
- **Gas Station Locations**: Sourced from Overpass Turbo, providing geographical coordinates of existing stations that could be upgraded to HRS.

## Methodologies

### Mathematical Model
- **Objective**: Maximize the coverage of vehicles by strategically upgrading existing gas stations to HRS, considering traffic intensity as a proxy for demand.
- **Constraints**: Includes distance limits between stations and traffic sensors, budget limitations, and non-overlapping coverage areas.

### Sensitivity Analysis
- **Total Investment Budget**: Explores the impact of different budget levels on the number and location of upgraded stations.
- **Maximum Distance**: Analyzes the effect of varying the coverage radius on station placement and overall network efficiency.

### Data Preprocessing
- **Traffic Data Cleaning**: Outliers and inconsistent entries were removed to ensure accurate and reliable input for the optimization model.
- **Distance Calculation**: Straight-line distances between gas stations and traffic sensors were calculated to determine potential coverage areas.

## Key Findings and Recommendations
The analysis suggests a staggered investment approach:
1. **Highways**: Focus initial investments on upgrading stations along highways, where larger coverage areas are feasible, maximizing cost-efficiency for long-distance transportation.
2. **Urban Areas**: Target high-demand urban areas with smaller coverage radii, ensuring sufficient HRS availability where traffic density is highest.

These strategies aim to balance coverage and investment costs, providing actionable insights for scaling HRS infrastructure efficiently.

## Tools and Libraries
- **Python**: Core programming language used for data processing and optimization.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Matplotlib**: Data visualization.
- **Gurobipy**: Optimization model solver.
