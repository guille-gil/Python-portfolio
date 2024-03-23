"""
Assignment DAPOM - Car Sharing Analysis case
Guillermo Gil de Avalle Bellido
"""

# Importing libraries. Some may appear unused due to the relevant code commented out.
from elasticsearch import Elasticsearch, helpers  # Helpers imported twice to ensure only one doc
from elasticsearch.helpers import scan, bulk
from loaders import ingest_json_file_into_elastic_index
from datetime import datetime
from gurobipy import Model, GRB
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import smopy
import json

# Connecting to Elasticsearch and defining initial variables
es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}]).options(ignore_status=[400, 405])
index_name = "car_sharing"
data_path = "C:/Users/guill/Desktop/DAPOM/Assignment/Assignment/request_data.json"


# PART 1: READ AND INGESTION
# INGESTING JSON FILE
# Define mapping for the ingestion
settings = {
    'settings': {
        "number_of_shards": 3
    },
    'mappings': {
        'properties': {
            'origin_datetime': {'type': 'date', "format": "yyyy-MM-dd HH:mm:ss"},
            'destination_datetime': {'type': 'date', "format": "yyyy-MM-dd HH:mm:ss"},
            'origin_location': {'type': 'geo_point'},
            'destination_location': {'type': 'geo_point'},
            'request_nr': {'type': 'integer'},
            'weekend': {'type': 'integer'}
        }
    }
}

# Create index and ingest data using loaders.py function
"""
es.indices.create(index=index_name, body=settings)
ingest_json_file_into_elastic_index(data_path, es, index_name, buffer_size=5000)
print("Index created and ingested")
"""


# Check first five lines to check whether ingestion is correct
"""
print("Sample documents:", es.search(index=index_name, body={"query": {"match_all": {}}, "size": 5}))
"""
# Commented because check was successful


# Determine the initial number of entries to compare with the number of entries after deleting the incorrect ones
es.indices.refresh(index=index_name)
print(f"Number of entries = {es.count(index=index_name)['count']}")

# CLEANING DATA
batch_size = 10000
batch = []

all_query = {
    "query": {
        "bool": {
            "must": {
                "match_all": {}
            }
        }
    }
}


# Function to calculate distance using origin lon/lat and destination lon/lat
def calculate_distance(origin, destination):
    # Using Geopy library
    result = geodesic(origin, destination)
    return result


# Function to swipe longs and lats
def swap_lon_lat(wrong_location):
    right_location = [wrong_location[1], wrong_location[0]]
    return right_location


# Function to calculate time difference in hours from origin & destination datetime
def calculate_time_difference_in_hours(origin, destination):
    time_difference = destination - origin
    time_difference_in_hours = time_difference.total_seconds() / 3600  # The number of seconds in an hour
    return time_difference_in_hours


# Function to calculate speed in kms using formula speed = distance / time.
def calculate_speed(distances, time):
    if time > 0:
        speed = distances / time
    else:
        speed = 0
    return speed


# Alter timeout for bulk operations (according to new practices of Elasticsearch)
es_with_options = es.options(request_timeout=30)

# Application - Commented out, due to successful outcome
"""
# Iterating over each instance and loading the main fields in variables
for hit in scan(client=es, index=index_name, query=all_query):
    origin_location = hit['_source']['origin_location']
    destination_location = hit['_source']['destination_location']
    origin_datetime = datetime.strptime(hit['_source']['origin_datetime'], "%Y-%m-%d %H:%M:%S")
    destination_datetime = datetime.strptime(hit['_source']['destination_datetime'], "%Y-%m-%d %H:%M:%S")

    # Check whether location and datetime are different. If so, set delete = 0, otherwise set delete = 1
    if origin_location != destination_location and origin_datetime != destination_datetime:

        # Swipe longs and lats
        origin_location = swap_lon_lat(origin_location)
        destination_location = swap_lon_lat(destination_location)

        # Calculate distance
        distance_km = calculate_distance(origin_location, destination_location).km

        # Calculate time difference in hours
        time_hours = calculate_time_difference_in_hours(origin_datetime, destination_datetime)

        # Calculate speed
        speed_kmh = calculate_speed(distance_km, time_hours)

        # Check whether speed is reasonable. If not, set delete = 1
        if 5 <= speed_kmh <= 50:  # For urban areas, speed typically ranges an avg of 5km/h to 50km/h
            # Update the document with distance, time, and speed, and set delete = 0
            update_request = {
                "_op_type": "update",
                "_index": index_name,
                "_id": hit['_id'],
                "doc": {
                    "distance_km": distance_km,
                    "time_hours": time_hours,
                    "speed_kmh": speed_kmh,
                    "demand": 1,
                    "delete": 0
                }
            }
            # Update request to batch
            batch.append(update_request)
        else:
            # Update the document with distance, time, and speed, and set delete = 1
            update_request = {
                "_op_type": "update",
                "_index": index_name,
                "_id": hit['_id'],
                "doc": {
                    "distance_km": distance_km,
                    "time_hours": time_hours,
                    "speed_kmh": speed_kmh,
                    "demand": 1,
                    "delete": 1
                }
            }
            # Add update request to batch
            batch.append(update_request)
            if len(batch) >= batch_size:
                bulk(client=es_with_options, actions=batch)
                batch = []
                print("Sent to elasticsearch")

if batch:
    bulk(client=es_with_options, actions=batch)

print("Distance, time, and speed calculations and updates completed.")

# Query to delete all element in which delete = 1
delete_query = {
  "query": {
    "term": {
      "delete": {
        "value": 1
      }
    }
  }
}

es.delete_by_query(index=index_name, body=delete_query, wait_for_completion=True)

# Determine the number of entries after deleting the incorrect ones
print(f"Number of entries = {es.count(index=index_name)['count']}")
"""

# PART 2: PLOTTING TIME SERIES OF DAILY DEMAND
# Define query for aggregation in order to plot time series of daily demand
aggregation_query = {
    "size": 0,
    "aggs": {
        "daily_demand": {
            "date_histogram": {
                "field": "origin_datetime",
                "calendar_interval": "day"
            }
        }
    }
}

# Execute the query for daily demand
aggregation_response = es.search(index=index_name, body=aggregation_query)

# Parse the response to get the data for plotting
dates = []
demand = []
for bucket in aggregation_response['aggregations']['daily_demand']['buckets']:
    dates.append(bucket['key_as_string'])
    demand.append(bucket['doc_count'])

# Convert dates to pandas datetime, which slices better and works better with matplotlib
dates = pd.to_datetime(dates)

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(dates, demand, marker='o', markersize=4, linestyle='-', color='b')
plt.xticks(rotation=45)  # Rotates x labels 45 degrees for better readability
plt.tight_layout()  # Adjust plots to give optimal spacing between each other
plt.title('Daily Demand Over the Past Year')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# PART 3: DEMAND FORECASTING
# VISUALIZE DEMAND IN HOURLY BASIS
# Query hourly data for weekdays
hourly_query = {
    "query": {
        "bool": {
            "must": [
                {"term": {"weekend": 0}}  # Only consider working days
            ]
        }
    },
    "aggs": {
        "hourly_demand": {
            "date_histogram": {
                "field": "origin_datetime",
                "calendar_interval": "hour",
                "min_doc_count": 1  # Making sure the instance has data to show
            },
            "aggs": {
                "demand": {"value_count": {"field": "request_nr"}}
            }
        }
    },
    "size": 0
}

# Execute the query
hourly_response = es.search(index=index_name, body=hourly_query)

# Process the results & add into new lists
hours = []
demands = []
for bucket in hourly_response['aggregations']['hourly_demand']['buckets']:
    hour = datetime.utcfromtimestamp(bucket['key'] / 1000).hour
    # Convert to UTC from UNIX time. Divide by 1000 to convert to seconds. Select only hours.
    demand = bucket['demand']['value']
    hours.append(hour)
    demands.append(demand)

# Create a DataFrame (long format) for easier manipulation
df_long_format = pd.DataFrame({'Hour': hours, 'Demand': demands})

# Group by hour and calculate mean, std and overall mean demand
stats = df_long_format.groupby('Hour')['Demand'].agg(['mean', 'std', 'min', 'max']).reset_index()
overall_mean_demand = stats['mean'].mean()

plt.figure(figsize=(14, 8))

# Plotting mean demand as bars
plt.bar(stats['Hour'], stats['mean'], color='skyblue', label='Mean Demand')

# Adding error bars for standard deviation directly on the bars
plt.errorbar(stats['Hour'], stats['mean'], yerr=stats['std'], fmt='none', ecolor='gray',
             capsize=5, label='Standard Deviation')

# Min and Max values as points on top of each bar
plt.scatter(stats['Hour'], stats['min'], color='red', marker='_', s=100, label='Min Demand', zorder=5)
plt.scatter(stats['Hour'], stats['max'], color='blue', marker='_', s=100, label='Max Demand', zorder=5)

# Overall mean demand as a horizontal line
plt.axhline(y=overall_mean_demand, color='green', linestyle='--', label='Overall Mean Demand', zorder=4)

plt.title('Hourly Demand on Working Days (with Mean, Standard Deviation, Min, and Max)')
plt.xlabel('Hour of the Day')
plt.ylabel('Demand')
plt.xticks(np.arange(stats['Hour'].min(), stats['Hour'].max() + 1, 1.0))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Generate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

# EXPECTED NUMBER OF REQUESTS ON A WORKING DAY
workday_query = {
    "size": 0,
    "query": {
        "bool": {
            "must": [
                {"term": {"weekend": 0}}  # Only consider working days
            ]
        }
    },
    "aggs": {
        "daily_demand": {
            "date_histogram": {
                "field": "origin_datetime",
                "calendar_interval": "day",
                "min_doc_count": 1  # Ensure we only count days with at least one request
            }
        }
    }
}

# Execute the query
workday_response = es.search(index=index_name, body=workday_query)

# Extract the daily demand counts
daily_demands = [bucket['doc_count'] for bucket in workday_response['aggregations']['daily_demand']['buckets']]

# Calculate the average daily demand
average_daily_demand = np.mean(daily_demands)

# Increase by 30% to account for unsatisfied demand
adjusted_average_daily_demand = round(average_daily_demand * 1.3, 0)

print(f"Average daily demand (adjusted for unsatisfied demand): {adjusted_average_daily_demand}")


# VISUALIZE A SAMPLE FROM 3(a) AND 3(b)
# Using average daily demand as the size of our sample, as calculated in 3b
sample_size = int(adjusted_average_daily_demand)

# Selecting a random sample of the data
random_sample_df = df_long_format.sample(n=sample_size, random_state=42)
# random_state=42 to ensure that the same sample is selected every time

# Increase the 'Demand' column values by 30%
random_sample_df['Adjusted_Demand'] = random_sample_df['Demand'] * 1.30

# Group by hour and calculate statistics on Adjusted_Demand
stats_random_df = random_sample_df.groupby('Hour')['Adjusted_Demand'].agg(['mean', 'std', 'min', 'max']).reset_index()
overall_mean_demand_adjusted_random_df = stats_random_df['mean'].mean()

plt.figure(figsize=(14, 8))

# Plotting mean adjusted demand as bars
plt.bar(stats_random_df['Hour'], stats_random_df['mean'], color='skyblue', label='Mean Adjusted Demand')

# Adding error bars for standard deviation of the adjusted demand
plt.errorbar(stats_random_df['Hour'], stats_random_df['mean'], yerr=stats_random_df['std'], fmt='none', ecolor='gray',
             capsize=5, label='Standard Deviation')

# Min and Max values as points on top of each bar for adjusted demand
plt.scatter(stats_random_df['Hour'], stats_random_df['min'], color='red', marker='_',
            s=100, label='Min Adjusted Demand', zorder=5)
plt.scatter(stats_random_df['Hour'], stats_random_df['max'], color='blue', marker='_',
            s=100, label='Max Adjusted Demand', zorder=5)

# Overall mean adjusted demand as a horizontal line
plt.axhline(y=overall_mean_demand_adjusted_random_df, color='green',
            linestyle='--', label='Overall Mean Adjusted Demand', zorder=4)

plt.title('Hourly Adjusted Demand on Working Days (Random Sample) - Mean, Standard Deviation, Min, and Max')
plt.xlabel('Hour of the Day')
plt.ylabel('Adjusted Demand')
plt.xticks(np.arange(stats_random_df['Hour'].min(), stats_random_df['Hour'].max() + 1, 1.0))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Generate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

# Print the number of instances in random_sample_df
print("Number of instances in random_sample_df:", len(random_sample_df))


# PART 4: OPTIMIZATION MODEL
# Load car data
cars_location_df = []
with open("C:/Users/guill/Desktop/DAPOM/Assignment/Assignment/car_locations.json", 'r') as file:
    for line in file:
        car = json.loads(line)
        # The geo-data for the first 200 lines is swapped. Cleaning it...
        if len(cars_location_df) < 200:
            car['start_location'] = [car['start_location'][1], car['start_location'][0]]
        cars_location_df.append(car)

# Load again required request_data.json from Elasticsearch
query = {
    "size": sample_size,
    "query": {
        "match_all": {}
    }
}

response = es.search(index=index_name, body=query)
requests_data_df = []

# Format and clean it
for hit in response['hits']['hits']:
    request = hit['_source']
    origin_location = tuple(request['origin_location'])
    destination_location = tuple(request['destination_location'])
    origin_location = swap_lon_lat(origin_location)
    destination_location = swap_lon_lat(destination_location)

    requests_data_df.append({
        "request_id": hit['_id'],
        "origin_location": origin_location,
        "destination_location": destination_location,
        "walking_distance": 0.4,  # Assuming 400 meters for all requests
        "profit": 0
    })

print(f"The sample size for optimization is: {len(requests_data_df)}")

# Define parameters
average_speed_kmh = 50
revenue_rate = 0.19
maximum_distance = 0.4

# Define as a list as it will receive more values later
profits = []
matched_assignments = []


# Pre-emptively populate profit to avoid errors (based on formula distance / speed)
for request in requests_data_df:
    distance_km = calculate_distance(request['origin_location'], request['destination_location']).kilometers
    travel_time_hours = distance_km / average_speed_kmh
    travel_time_minutes = np.ceil(travel_time_hours * 60)
    request['profit'] = travel_time_minutes * revenue_rate


# Function to check the pre-process compatibility and distance for any given w
def calculate_compatibility_and_distance(car_data, request_data, walking_distances):
    compatibility_for_walking_distances = {w: set() for w in walking_distances}
    distances = {}  # This will store the calculated distances to avoid recalculating

    for car in car_data:
        car_id = car['car_id']
        car_start_location = tuple(car['start_location'])

        for request in request_data:
            request_id = request['request_id']
            request_origin_location = tuple(request['origin_location'])

            # Key for distances dictionary
            distance_key = (car_id, request_id)

            # Calculate distance if not already done
            if distance_key not in distances:
                distances[distance_key] = calculate_distance(car_start_location, request_origin_location).km

            # Check compatibility for each walking distance
            for w in walking_distances:
                if distances[distance_key] <= w:
                    compatibility_for_walking_distances[w].add((car_id, request_id))
    return compatibility_for_walking_distances, distances


# Function that calculates maximization model for any given w
def optimize_model(car_data, request_data, w, compatibility_for_walking_distances):
    model = Model("Car_sharing")

    num_cars = len(car_data)
    num_requests = len(request_data)

    x = model.addVars(num_cars, num_requests, vtype=GRB.BINARY, name="assign")

    model.setObjective(
        sum(x[c, j] * request_data[j]['profit']
            for c, car in enumerate(car_data)
            for j, request in enumerate(request_data)
            if (car['car_id'], request['request_id']) in compatibility_for_walking_distances[w]),
        GRB.MAXIMIZE)

    for j, request in enumerate(request_data):
        model.addConstr(sum(x[c, j] for c, car in enumerate(car_data)
                if (car['car_id'], request['request_id']) in compatibility_for_walking_distances[w]) <= 1)

    for c, car in enumerate(car_data):
        model.addConstr(
            sum(x[c, j] for j, request in enumerate(request_data)
                if (car['car_id'], request['request_id']) in compatibility_for_walking_distances[w]) <= 1)

    model.optimize()
    return model


# Precompute compatibility and distances for only 0.4km
compatibility_for_walking_distances, distances = (calculate_compatibility_and_distance(cars_location_df, requests_data_df, [0.4]))

# Processing the model only for 0.4km
model = optimize_model(cars_location_df, requests_data_df, maximum_distance, compatibility_for_walking_distances)
if model.status == GRB.OPTIMAL:
    profits.append(model.getObjective().getValue())

    matched_assignments = []
    for c, car in enumerate(cars_location_df):
        for j, request in enumerate(requests_data_df):
            if (car['car_id'], request['request_id']) in compatibility_for_walking_distances[0.4]:
                if model.getVarByName(f"assign[{c},{j}]").X > 0.5:
                    print(f"Request {request['request_id']} assigned to car {car['car_id']} in 0.4km walking distance")
                    matched_assignments.append((c, j))

    print(f"The number of matched requests for 0.4 km walking distance is {len(matched_assignments)}")
else:
    profits.append(0)
    print(f"Optimization failed for 0.4km")

# Create lists for matched_assignments for printing
matched_lats = []
matched_lons = []

for car_index, request_index in matched_assignments:
    request = requests_data_df[request_index]
    matched_lats.append(request['origin_location'][0])
    matched_lons.append(request['origin_location'][1])

# Map visualization for matched requests and their connections to cars
matched_map = smopy.Map((min(matched_lats), min(matched_lons), max(matched_lats), max(matched_lons)), z=12)
ax = matched_map.show_mpl(figsize=(10, 10))

# Plotting cars
for c, car in enumerate(cars_location_df):
    cx, cy = matched_map.to_pixels(car['start_location'][0], car['start_location'][1])
    ax.plot(cx, cy, 'r^', markersize=8, markeredgecolor='k', label='Cars' if c == 0 else "")

# Plotting matched requests and drawing lines
for c, j in matched_assignments:
    request = requests_data_df[j]
    ox, oy = matched_map.to_pixels(request['origin_location'][0], request['origin_location'][1])
    dx, dy = matched_map.to_pixels(request['destination_location'][0], request['destination_location'][1])
    cx, cy = matched_map.to_pixels(cars_location_df[c]['start_location'][0], cars_location_df[c]['start_location'][1])
    ax.plot(ox, oy, 'go', markersize=5, markeredgecolor='k', label='Matched Request Origins' if c == 0 else "")
    ax.plot(dx, dy, 'bo', markersize=5, markeredgecolor='k', label='Matched Request Destinations' if c == 0 else "")
    ax.plot([cx, ox], [cy, oy], '-', color='k')  # Line from car to request origin

# Adding a title and legend
ax.set_title("Matched Requests and Their Assigned Cars")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.show()

# Extract request indices from matched_assignments
matched_request_indices = {request_index for _, request_index in matched_assignments}

# Calculate unmatched_requests_indices by subtracting matched_request_indices from the set of all request indices
unmatched_requests_indices = set(range(len(requests_data_df))) - matched_request_indices

# Now you can proceed with calculating unmatched_lats and unmatched_lons
unmatched_lats = [requests_data_df[j]['origin_location'][0] for j in unmatched_requests_indices]
unmatched_lons = [requests_data_df[j]['origin_location'][1] for j in unmatched_requests_indices]

# Check if there are any unmatched requests to plot
if unmatched_lats and unmatched_lons:
    unmatched_map = smopy.Map((min(unmatched_lats), min(unmatched_lons), max(unmatched_lats), max(unmatched_lons)), z=12)
    ax = unmatched_map.show_mpl(figsize=(10, 10))

    # Plot the origins of unmatched requests
    for lat, lon in zip(unmatched_lats, unmatched_lons):
        ux, uy = unmatched_map.to_pixels(lat, lon)
        ax.plot(ux, uy, 'go', markersize=5, markeredgecolor='k',
                label='Unmatched Request' if lat == unmatched_lats[0] else "")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title("Unmatched Requests (Only Origins)")
    plt.show()


# PART 5 - CORRELATION BETWEEN WALKING DISTANCE AND PROFIT
# Define walking distances to iterate over
walking_distances = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                     0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Restart profits
profits = []

# Precompute compatibility and distances
compatibility_for_walking_distances, distances = calculate_compatibility_and_distance(cars_location_df, requests_data_df, walking_distances)

# Execute modeling and calculate different profits
for w in walking_distances:
    model = optimize_model(cars_location_df, requests_data_df, w, compatibility_for_walking_distances)
    if model.status == GRB.OPTIMAL:
        profits.append(model.getObjective().getValue())
        matched_assignments = []
        for c, car in enumerate(cars_location_df):
            for j, request in enumerate(requests_data_df):
                if (car['car_id'], request['request_id']) in compatibility_for_walking_distances[w]:
                    if model.getVarByName(f"assign[{c},{j}]").X > 0.5:
                        print(f"Request {request['request_id']} assigned to car {car['car_id']} in {w} km walking distance")
                        matched_assignments.append((c, j))

        print(f"The number of matched requests in {w} km walking distance is {len(matched_assignments)}")
    else:
        profits.append(0)
        print(f"Optimization failed for walking distance {w}")

# Calculate the overall profit variability as the standard deviation of the given profits
profit_variability = np.std(profits)

# Calculate upper and lower bounds for the shaded region representing variability
upper_bound = [profit + profit_variability for profit in profits]
lower_bound = [profit - profit_variability for profit in profits]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(walking_distances, profits, marker='o', linestyle='-', color='blue', label='Total Profit')
plt.fill_between(walking_distances, lower_bound, upper_bound, color='blue', alpha=0.1, label='Profit Variability')

# Adding enhancements for visual appeal
plt.title('Total Profit vs. Walking Distance with Variability', fontsize=16)
plt.xlabel('Walking Distance (km)', fontsize=14)
plt.ylabel('Total Profit', fontsize=14)
plt.xticks(walking_distances, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping

# Add a vertical line at 0.4 km
plt.axvline(x=0.4, color='red', linestyle='--', label="Maximum walking distance 'W'")
plt.text(0.4, max(upper_bound), "Maximum walking distance 'W'", horizontalalignment='right', color='red', fontsize=12)

plt.show()
