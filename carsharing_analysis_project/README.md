# Car Sharing Company Optimization in Groningen

## Overview
This project focuses on addressing the operational challenges faced by a car-sharing service in Groningen, Netherlands. The primary goals are to improve profitability and customer satisfaction through data analysis and optimization techniques. We leveraged a year's worth of detailed car and customer request data to uncover demand patterns, optimize fleet allocation, and propose actionable insights to enhance service efficiency.

## Folder Structure
- `main.py`: The main Jupyter notebook containing the full analysis, from data ingestion and cleaning to demand pattern analysis and optimization.
- `loaders.py`: A support script for bulk ingestion of datasets into Elasticsearch, facilitating efficient data handling and preparation.
- `PDF report`: A detailed written report explaining the coding logic, analysis results, and strategic recommendations derived from the study.

## Datasets
The project utilizes two primary datasets:
- `car_locations.json`: Contains locations of cars within the service area. Note: The first 200 entries had their latitude and longitude coordinates reversed.
- `request_data.json`: Documents a year's worth of customer requests, with a similar issue of reversed lat-long coordinates in all entries.

Both datasets underwent a thorough cleaning process to correct the coordinate inversions and other inconsistencies, ensuring the reliability of subsequent analyses.

## Methodologies
### Data Ingestion and Cleaning
- Employed Elasticsearch for efficient handling of large datasets.
- Developed a cleaning pipeline in Python to correct coordinate inversions and filter out implausible data points.

### Demand Pattern Analysis
- Utilized Elasticsearch aggregations and Pandas for exploratory data analysis.
- Identified seasonal trends and daily demand fluctuations to inform operational adjustments.

### Demand Forecasting Analysis
- Forecasted future car demand using Elasticsearch and Python analysis tools.
- Highlighted key demand peaks during working days, providing insights for fleet allocation.

### Optimization Problem
- Formulated an optimization model using Gurobi to maximize profits by efficiently matching cars with customer requests.
- Explored the impact of varying walking distance constraints on service coverage and profitability.

## Visualization
- Employed Matplotlib for the creation of informative visualizations depicting demand patterns, forecasting, and optimization results.
- Used Smopy for geographical plotting of car locations and customer request origins/destinations.

## Conclusions and Recommendations
The analysis revealed significant demand seasonality and identified optimal times for fleet allocation. The optimization model highlighted the limitations of the current operational strategy, prompting recommendations for dynamic car positioning and targeted service enhancements to improve efficiency and customer satisfaction.

## Tools and Libraries
- Elasticsearch: NoSQL database for scalable data storage and analysis.
- Pandas: Data manipulation and analysis.
- Matplotlib: Data visualization.
- Smopy: Mapping tool for visualizing geographical data.
- Gurobi: Advanced optimization software.

For a more detailed exploration of the methodologies, findings, and code, please refer to the included PDF report.
