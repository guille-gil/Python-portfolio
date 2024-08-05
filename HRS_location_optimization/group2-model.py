import pandas as pd
import geopandas as gpd
from pyproj import Geod
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import folium
import matplotlib.pyplot as plt

# Load sensor data from Excel
sensor_data = pd.read_excel('final.xlsx')
sensor_data = sensor_data[['id_sensor', 'lat', 'lon', 'avg_intensity']]

# Load station data from GeoJSON
station_data = gpd.read_file('stations_data.geojson')
station_data['lat'] = station_data.geometry.y
station_data['lon'] = station_data.geometry.x

# Filter sensor data based on intensity percentile
percentile = sensor_data['avg_intensity'].quantile(0.25)
filtered_data = sensor_data[sensor_data['avg_intensity'] >= percentile]

# Calculate the total number of sensors activated (sum of avg_intensity)
total_number_sensors_activated = filtered_data['avg_intensity'].sum()

# Setup geodetic calculations
geod = Geod(ellps="WGS84")

# Vectorized calculation of distances
def calculate_distances(station_data, sensor_data):
    station_lats, station_lons = station_data['lat'].values, station_data['lon'].values
    sensor_lats, sensor_lons = sensor_data['lat'].values, sensor_data['lon'].values

    lon1, lon2 = np.meshgrid(station_lons, sensor_lons, indexing='ij')
    lat1, lat2 = np.meshgrid(station_lats, sensor_lats, indexing='ij')

    _, _, distances = geod.inv(lon1, lat1, lon2, lat2)
    return distances

# Compute distances
distance_matrix = calculate_distances(station_data, filtered_data)

def build_and_optimize_stations(station_data, sensor_data, distances, installation_cost, max_distance):
    model = gp.Model("Station_Placement")

    stations = station_data.index.tolist()
    sensors = sensor_data.index.tolist()
    x = model.addVars(stations, vtype=GRB.BINARY, name="x")  # Whether to place a station
    z = model.addVars(stations, sensors, vtype=GRB.BINARY, name="z")  # Station-sensor assignment

    # Precompute y_ik as a parameter
    y = {(i, k): int(distances[i, k] <= max_distance) for i in range(distances.shape[0]) for k in range(distances.shape[1])}

    # Objective: Maximize total intensity served
    model.setObjective(gp.quicksum(z[i, k] * sensor_data.at[k, 'avg_intensity'] for i in stations for k in sensors), GRB.MAXIMIZE)

    # Constraints
    for k in sensors:
        model.addConstr(gp.quicksum(z[i, k] for i in stations) <= 1, f"UniqueAssignment_{k}")

    for i in stations:
        for k in sensors:
            if (i, k) in y:
                model.addConstr(z[i, k] <= x[i] * y[i, k], f"Service_{i}_{k}")

    budget_constr = model.addConstr(gp.quicksum(installation_cost * x[i] for i in stations) <= GRB.INFINITY, "Budget")

    model.update()
    return model, x, z, budget_constr

# Parameters
installation_cost = 3_850_000  # Fixed installation cost
budgets = np.linspace(20_000_000, 100_000_000, 10)  # Different budget scenarios
max_distances = [1000, 2500, 5000, 7500, 10000]  # Different max distances for sensitivity analysis

assert len(station_data) == distance_matrix.shape[0], "Mismatch between stations and distance matrix rows"
assert len(filtered_data) == distance_matrix.shape[1], "Mismatch between sensors and distance matrix columns"

sensitivity_results = []

# Iterate over different max_distance values
for max_distance in max_distances:
    print(f"Testing with max_distance: {max_distance}")

    # Build model
    model, x, z, budget_constr = build_and_optimize_stations(station_data, filtered_data, distance_matrix, installation_cost, max_distance)

    # Iterate over budgets and optimize
    for budget in budgets:
        print(f"Testing with budget: {budget}")
        budget_constr.RHS = budget
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print(f"\nBudget: {budget}")
            print(f"Total cars served: {model.objVal}")
            selected_stations = [i for i in station_data.index if x[i].X > 0.5]
            results = (max_distance, budget, model.objVal, selected_stations)
            sensitivity_results.append(results)
            print("Selected stations and their total car coverage:")
            sensors = filtered_data.index.tolist()
            for station in selected_stations:
                lat, lon = station_data.loc[station, ['lat', 'lon']]
                total_cars = sum(z[station, k].X * sensor_data.at[k, 'avg_intensity'] for k in sensors)
                print(f"Station at ({lat}, {lon}) serves {total_cars} cars")
        else:
            print(f"No feasible solution found for budget {budget}")

# Create maps for each max_distance and budget level
for max_distance, budget, num_vehicles_covered, selected_stations in sensitivity_results:
    budget_map = folium.Map(location=[station_data['lat'].mean(), station_data['lon'].mean()], zoom_start=12)

    for station in selected_stations:
        lat, lon = station_data.loc[station, ['lat', 'lon']]
        total_cars = sum(z[station, k].X * sensor_data.at[k, 'avg_intensity'] for k in sensors)

        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color='red', icon='info-sign'),
            popup=f"Station ID: {station}, Total cars: {total_cars}"
        ).add_to(budget_map)

        folium.Circle(
            location=(lat, lon),
            radius=max_distance,
            color='grey',
            fill=True,
            fill_color='grey',
            fill_opacity=0.2
        ).add_to(budget_map)

        # Add sensors within the radius
        for k in filtered_data.index:
            sensor_lat = filtered_data.at[k, 'lat']
            sensor_lon = filtered_data.at[k, 'lon']
            distance = geod.inv(lon, lat, sensor_lon, sensor_lat)[2]
            if distance <= max_distance:
                folium.CircleMarker(
                    location=(sensor_lat, sensor_lon),
                    radius=3,  # small radius for the dots
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6,
                    popup=f"Sensor ID: {k}, Intensity: {filtered_data.at[k, 'avg_intensity']}"
                ).add_to(budget_map)

    budget_map.save(f'station_placement_map_max_distance_{max_distance}_budget_{budget}.html')

# Print the average total cars covered per hour, per day, and the number of stations upgraded for each max_distance and budget level
print("\nSummary of Average Total Cars Covered and Number of Stations Upgraded:")

results_data = []

for max_distance, budget, num_vehicles_covered, selected_stations in sensitivity_results:
    total_cars_per_hour = num_vehicles_covered
    coverage = 100 * total_cars_per_hour / total_number_sensors_activated  # Calculate coverage as a percentage
    total_cars_per_day = total_cars_per_hour * 24
    num_stations_upgraded = len(selected_stations)
    avg_cost_per_car_served = budget / total_cars_per_day  # Calculate average cost per car served per day
    results_data.append([max_distance, budget, total_cars_per_hour, round(coverage, 3), total_cars_per_day, num_stations_upgraded, avg_cost_per_car_served])
    print(f"Max Distance: {max_distance}, Budget: {budget}, Total cars per hour: {total_cars_per_hour}, Coverage: {round(coverage, 3)}%, Total cars per day: {total_cars_per_day}, Number of stations upgraded: {num_stations_upgraded}, Avg cost per car served: {avg_cost_per_car_served}")

# Convert results to DataFrame
results_df = pd.DataFrame(results_data, columns=['Max Distance', 'Budget', 'Total cars per hour', 'Coverage (%)', 'Total cars per day', 'Number of stations upgraded', 'Avg cost per car served'])

# Save to Excel
results_df.to_excel('sensitivity_analysis_results.xlsx', index=False)

# Plot a line chart of the average cost per car served for each level of investment and max_distance
plt.figure(figsize=(10, 6))
for max_distance in max_distances:
    budget_values = [result[1] / 1_000_000 for result in results_df.values if result[0] == max_distance]  # Scale the budget values to millions
    avg_cost_values = [result[6] for result in results_df.values if result[0] == max_distance]  # Extract the average cost per car served

    plt.plot(budget_values, avg_cost_values, marker='o', linestyle='-', label=f'Max Distance {max_distance}m')

plt.title('Average Cost per Car Served for Each Level of Investment and Max Distance')
plt.xlabel('Investment (Budget in Millions)')
plt.ylabel('Average Cost per Car Served')
plt.grid(True)
plt.legend()
plt.show()

# Plot a line chart of the average car coverage per day for each level of investment and max_distance
plt.figure(figsize=(10, 6))
for max_distance in max_distances:
    budget_values = [result[1] / 1_000_000 for result in results_df.values if result[0] == max_distance]  # Scale the budget values to millions
    coverage_values = [result[3] for result in results_df.values if result[0] == max_distance]  # Extract the pre-calculated coverage values

    plt.plot(budget_values, coverage_values, marker='o', linestyle='-', label=f'Max Distance {max_distance}m')

plt.title('Percentage of Coverage per Day for Each Level of Investment and Max Distance')
plt.xlabel('Investment (Budget in Millions)')
plt.ylabel('Percentage of Coverage per Day (%)')
plt.grid(True)
plt.legend()
plt.show()
