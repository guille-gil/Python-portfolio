# Asset Management and Maintenance Optimization

## Overview
This project delves into the optimization of maintenance policies for three different machines using historical maintenance data. By employing statistical analysis techniques such as Kaplan-Meier and Weibull, alongside simulations based on Condition-Based Monitoring (CBM) data, we aim to identify the most cost-effective maintenance strategy for each machine. The project's objective is to enhance the reliability and cost-efficiency of asset management practices.

## Folder Structure
- `Tool.py`: The primary Python script containing all data import, analysis, simulation, and policy comparison logic.

_Theoretical folders, if we wanted to make the tool functional_
- `maintenance_data/`: Directory containing the raw maintenance datasets for the three machines.
- `simulation_results/`: Generated reports and graphs from the simulation runs, comparing different maintenance thresholds.
- `policy_comparison.xlsx`: Excel spreadsheet summarizing the unitary costs for each maintenance policy and indicating the recommended strategies.

## Datasets
The project analyzes maintenance records for three distinct machines, focusing on failure modes, maintenance actions, and intervals. CBM data, where available, was critically used to simulate the effect of varying maintenance thresholds on overall machine reliability and maintenance cost.

## Methodologies
### Statistical Analysis
- **Kaplan-Meier:** Used to estimate the survival functions from lifetime data, providing insights into the reliability of machines over time.
- **Weibull Analysis:** Employed to model the time-to-failure data, helping to understand the probability of failure as a function of time.

### Simulation
- A simulation approach based on CBM data to evaluate different maintenance thresholds, determining the optimal point for conducting maintenance activities to minimize cost while maximizing reliability.

### Policy Comparison
- Comparative analysis of unitary costs associated with different maintenance policies to select the most cost-effective approach.

## Key Findings and Recommendations
The analysis and simulations reveal significant insights into maintenance strategies, highlighting the effectiveness of condition-based maintenance thresholds. Recommendations are provided based on a detailed cost-benefit analysis, guiding towards the optimal maintenance policy, cost, and maintenance time for each machine.

## Tools and Libraries
- Python: For data manipulation, analysis, and simulation.
- Pandas: Data analysis and manipulation.
- NumPy: Numerical computing.
- Matplotlib: Visualization of the findings.
- Lifelines: Implementation of Kaplan-Meier and Weibull analysis.
