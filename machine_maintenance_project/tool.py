import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
np.random.seed(0)

student_nr = "s5787084"
data_path = "C:/Users/guill/Desktop/AM Assignment 3/s5787084/"


def data_preparation(data):
    # Create a copy of the data to avoid altering original one
    copy_data = data

    # Secondly creating a column 'Duration', that calculates the difference of time
    copy_data['Duration'] = round(copy_data['Time'].diff(1), 2)

    # However, the first value now will reflect NaN, so adjusting that
    copy_data.loc[0, 'Duration'] = copy_data.loc[0, 'Time']

    # Add a temporary helper column for sorting 'Event' with 'failure' prioritized over 'PM'
    # Assign lower numerical values to 'failure' so it comes first
    copy_data['EventPriority'] = copy_data['Event'].map({'failure': 1, 'PM': 2})

    # Now we sort by duration and then by 'EventPriority' to prioritize 'failure' over 'PM'
    copy_data = copy_data.sort_values(by=['Duration', 'EventPriority'], ignore_index=True)

    # Finally dropping the 'EventPriority' column, as it's no longer needed
    copy_data.drop(columns=['EventPriority'], inplace=True)

    return copy_data


def create_kaplanmeier_data(data):
    # Create a copy of the data to avoid altering original one
    copy_data = data

    # Create rows count variable to save processing time
    num_rows = len(data)

    # Initialize probabilities equally
    copy_data['Probability'] = 1 / num_rows  # Distribute initial probability amongst rows
    copy_data['Reliability'] = 1.0  # Start with full reliability

    # Distribute 'PM' probabilities to subsequent rows
    for i in range(num_rows):
        if copy_data.loc[i, 'Event'] == 'PM':
            rows_after = num_rows - i - 1
            if rows_after > 0:
                value_to_add = copy_data.loc[i, 'Probability'] / rows_after
                for j in range(i + 1, num_rows):
                    copy_data.loc[j, 'Probability'] += value_to_add
            copy_data.loc[i, 'Probability'] = 0  # Set 'PM' probability to 0

    # Adjust reliability cumulatively for 'failure' events
    for i in range(1, num_rows):  # Start from the second row
        # Calculate cumulative sum up to point 'i' and assign back
        copy_data.loc[i, "Reliability"] = 1 - data.loc[:i, 'Probability'].cumsum().iloc[-1]

    return copy_data


def meantimebetweenfailure_KM(data):
    copy_data = data
    MTBF_km = 0
    for i in range(len(copy_data)):
        MTBF_km += copy_data.loc[i, "Duration"] * copy_data.loc[i, "Probability"]
    return MTBF_km


def fit_weibull_distribution(data):
    l_range = np.linspace(start=1, stop=35, num=35)  # Lambda values from 1 to 35
    k_range = np.linspace(start=0.1, stop=3.5, num=35)  # Kappa values from 0.1 to 3.5

    weibull_results = []  # Prepare a list to hold results

    for l in l_range:
        for k in k_range:
            weibull_values = []  # List to store values for each observation

            for index, row in data.iterrows():
                t = row['Duration']  # Duration for the observation

                # Ensure duration is non-negative
                if t < 0:
                    raise ValueError("Invalid duration encountered. Duration must be >= 0.")

                # Determine if the observation is censored based on the 'Event' column
                censored, uncensored = [(row['Event'] == 'PM'), (row['Event'] == 'failure')]

                # Adjust the calculation based on the censored status
                if censored:
                    formula_result = np.exp(- (t / l) ** k)
                elif uncensored:
                    formula_result = (k / l) * (t / l) ** (k - 1) * np.exp(- (t / l) ** k)
                else:
                    formula_result = -np.inf  # Account for np infinitives (just in case)

                # Append the calculated value to the list
                weibull_values.append(formula_result)

            # Sum of log of these values for 'Loglikelihood_sum'
            loglikelihood_sum = np.sum(np.log(np.maximum(weibull_values, 1e-100)))  # Avoid log(0)

            # Append results for this (lambda, kappa) pair
            weibull_results.append([l, k] + weibull_values + [loglikelihood_sum])

    # Column names for the DataFrame
    weibull_columns = ['lambda', 'kappa'] + [f'Observation {i + 1}' for i in range(len(data))] + ['Loglikelihood_sum']

    # Create the DataFrame
    weibull_df = pd.DataFrame(weibull_results, columns=weibull_columns)

    """
    # If need to see the columns, uncomment here.
    print(weibull_df)
    """

    # Find the chosen lambda/kappa pair using 'idxmax'
    chosen_pair = weibull_df['Loglikelihood_sum'].idxmax()

    # Retrieve the lambda and kappa values for the best pair
    chosen_l = weibull_df.loc[chosen_pair, 'lambda']
    chosen_k = weibull_df.loc[chosen_pair, 'kappa']

    # Return the best lambda and kappa values
    return chosen_l, chosen_k


def meantimebetweenfailure_weibull(lam, kap):
    MTBF_weibull = lam * math.gamma(1 + 1 / kap)
    return MTBF_weibull


def create_weibull_curve_data(data, lam, kap):
    # Determine the range of t from 0 to the largest duration in the dataset
    max_duration = data['Duration'].max()
    t_values = np.linspace(0, max_duration, 1000)  # Generate 1000 points between 0 and max_duration

    # Calculate R(t) for each t using the Weibull reliability function
    R_t = np.exp(-((t_values / lam) ** kap))

    # Create a DataFrame to hold t and R(t)
    weibull_curve_data = pd.DataFrame({'t': t_values, 'R(t)': R_t})

    return weibull_curve_data


def visualization(KM_data, weibull_curve_data, machine):
    plt.figure(figsize=(10, 6))

    # Plot Kaplan-Meier Estimate as a step function
    plt.step(KM_data['Duration'], KM_data['Reliability'], label='Kaplan-Meier Estimate', where='post')

    # Plot Weibull Curve as a red line
    plt.plot(weibull_curve_data['t'], weibull_curve_data['R(t)'], 'r-', label='Weibull Reliability Curve')

    plt.xlabel('Time')
    plt.ylabel('Reliability')
    plt.title(f'Reliability Function for Machine {machine}')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_cost_data(prepared_data, l, k, PM_cost, CM_cost, machine_name):
    # Define F(t) and R(t) for Weibull distribution using 'lambda' for conciseness
    F = lambda t: 1 - np.exp(-(t / l) ** k)
    R = lambda t: np.exp(-(t / l) ** k)

    # Initialize DataFrame
    t_max = prepared_data['Duration'].max()
    ts = np.arange(0.01, t_max, 0.01)

    # Calculate R(t) and F(t) for all t values
    R_ts = R(ts)
    F_ts = F(ts)

    # Vectorized calculation for mean cycle length using 'cumsum', to perform operation in the whole array (faster)
    # Delta t is 0.01 as per ts definition
    delta_t = 0.01
    mean_cycle_length = np.cumsum(R_ts) * delta_t

    # Create a dataframe to store all results
    cost_df = pd.DataFrame({'t': ts, 'R(t)': R_ts, 'F(t)': F_ts, 'Mean Cycle Length': mean_cycle_length})

    # Calculate Cost per Cycle and Cost rate
    cost_df['Cost per Cycle'] = CM_cost * cost_df['F(t)'] + PM_cost * cost_df['R(t)']
    cost_df['Cost rate(t)'] = cost_df['Cost per Cycle'] / cost_df['Mean Cycle Length']

    # Plot, excluding smaller t that depicts high cost rates
    plotting_sample_t = cost_df[cost_df['t'] > 0.1]
    plt.figure(figsize=(10, 6))
    plt.plot(plotting_sample_t['t'], plotting_sample_t['Cost rate(t)'], label='Cost Rate')
    plt.xlabel('Maintenance Age (t)')
    plt.ylabel('Cost Rate')
    plt.title(f'Cost Rate vs. Maintenance Age for {machine_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate optimal age & cost rate using 'idxmin'
    optimal_row = cost_df.loc[cost_df['Cost rate(t)'].idxmin()]
    optimal_age = optimal_row['t']
    optimal_cost_rate = optimal_row['Cost rate(t)']

    return optimal_age, optimal_cost_rate


def CBM_data_preparation(data):
    # Create a copy of the data to avoid altering original one
    copy_data = data

    # Secondly creating a column 'Duration', that calculates the difference of time
    copy_data['Increments'] = copy_data['Condition'].diff(1)

    # Drop the first row first
    # The first column will not be 0 as in the case example. However, it is following the slides/professor instructions.
    copy_data = copy_data.iloc[1:]

    # Then drop rows with negative Increments and reset index
    copy_data = copy_data[copy_data['Increments'] >= 0].reset_index(drop=True)

    return copy_data


def CBM_create_simulations(data, failure_level, threshold):
    num_simulations = 1500  # Yield a good balance between smooth graph and approx 5 minutes running time (in my laptop)
    simulations_df = pd.DataFrame(columns=["Duration", "Event"])

    # Convert the increments to a list to facilitate random selection without repetition
    increments_list = list(data['Increments'].values)

    # Indexes of all available increments (only way to avoid repetition if two numbers are equal)
    total_indexes = list(range(len(increments_list)))

    for sim in range(num_simulations):
        # Start condition, time, and a list to track indexes and avoid duplication of chosen increments by indexes
        condition = 0
        time = 0
        chosen_indexes = []

        # Start main simulation loop after applying by applying the first increment
        while condition < failure_level:
            # Update available indexes to exclude those already chosen. Break if no more indexes
            available_indexes = [index for index in total_indexes if index not in chosen_indexes]
            if not available_indexes:
                break

            # Choose another increment out of the available. Increase condition and time
            chosen_index = np.random.choice(available_indexes)
            chosen_indexes.append(chosen_index)
            increment = increments_list[chosen_index]
            condition += increment
            time += 1

            # Check the condition after each increment
            if condition >= failure_level:
                # If condition met, record time and end simulation in failure
                simulations_df.loc[len(simulations_df)] = [time, "failure"]
                break
            elif condition >= threshold and condition < failure_level:
                # If condition met, record time and end simulation in PM
                simulations_df.loc[len(simulations_df)] = [time, "PM"]
                break

    return simulations_df


def CBM_analyze_costs(simulation_data, PM_cost, CM_cost):
    # Count the number of preventive and corrective maintenance events in simulations
    PM_events = len(simulation_data[simulation_data['Event'] == 'PM'])
    CM_events = len(simulation_data[simulation_data['Event'] == 'failure'])  # Assuming 'failure' indicates CM
    total_events = PM_events + CM_events

    # Calculate Mean Cost Per Cycle applying notes formula
    mean_cost_per_cycle = (PM_cost * PM_events / total_events) + (CM_cost * CM_events / total_events)

    # Calculate mean cycle length using formula
    mean_cycle_length = simulation_data['Duration'].mean()

    # Calculate CBM cost rate
    cost_rate = mean_cost_per_cycle / mean_cycle_length

    return cost_rate


def CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name):
    # Define the range of thresholds to evaluate
    threshold_range = np.arange(1, int(failure_level + 1), 1)  # To the nearest integer value

    # Dataframe to store thresholds and their corresponding cost rates
    cost_data_df = pd.DataFrame(columns=['Threshold', 'Cost Rate'])

    for threshold in threshold_range:
        # Perform simulations for each threshold within range
        simulation_data = CBM_create_simulations(prepared_condition_data, failure_level, threshold)

        # Evaluate the cost rate for each threshold within range
        cost_rate = CBM_analyze_costs(simulation_data, PM_cost, CM_cost)

        # Append the results to the dataframe using .loc
        next_index = len(cost_data_df)
        cost_data_df.loc[next_index] = {'Threshold': threshold, 'Cost Rate': cost_rate}

    # Plot the cost rates for maintenance thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(cost_data_df['Threshold'], cost_data_df['Cost Rate'], marker='o', linestyle='-', color='black')
    plt.title(f'Cost Rate vs. Maintenance Threshold for Machine {machine_name}')
    plt.xlabel('Maintenance Threshold')
    plt.ylabel('Cost Rate')
    plt.grid(True)
    plt.show()

    # Determine the optimal maintenance threshold and the corresponding cost rate
    optimal_index = cost_data_df['Cost Rate'].idxmin()
    CBM_threshold = cost_data_df.at[optimal_index, 'Threshold']
    CBM_cost_rate = cost_data_df.at[optimal_index, 'Cost Rate']

    return CBM_threshold, CBM_cost_rate


# Extra: Adding additional function to calculate pure corrective cost rate.
def cm_cost_rate_calculation(MTBF_KM, MTBF_Weibull, CM_cost):
    # Cost rate for CM as calculated on Week 5 Age-Based slides
    average_mtbf = (MTBF_KM + MTBF_Weibull) / 2
    optimal_cost_rate = CM_cost / average_mtbf
    return optimal_cost_rate


# Extra: Formula to report if preferred policy is TBM
def report_tbm(machine_name, best_cost_rate, cm_cost_rate):
    print(f"The optimal maintenance policy for Machine {machine_name} is Time-Based")
    print(f'The best cost rate for Machine {machine_name} is', round(best_cost_rate, 2))
    savings_tbm_cm = cm_cost_rate - best_cost_rate
    print(f"The savings against a purely corrective maintenance policy per time unit for Machine {machine_name} are:", round(savings_tbm_cm, 2))
    return


# Extra: Formula to report if preferred policy is CBM
def report_cbm(machine_name, CBM_cost_rate, best_cost_rate, cm_cost_rate):
    print(f"The optimal maintenance policy for Machine {machine_name} is Condition-Based")
    print(f"The best cost rate for Machine {machine_name} is", round(CBM_cost_rate, 2))
    savings_cbm_tm = best_cost_rate - CBM_cost_rate
    savings_cbm_cm = cm_cost_rate - CBM_cost_rate
    print(f"The savings against a purely corrective maintenance policy per time unit for Machine {machine_name} are:", round(savings_cbm_tm, 2))
    print(f"The savings against a time-based maintenance policy per time unit for Machine {machine_name} are:", round(savings_cbm_cm, 2))
    return


# Extra: Formula to report if preferred policy is CM
def report_cm(machine_name, cm_cost_rate):
    print(f"The optimal maintenance policy for Machine {machine_name} is Corrective")
    print(f"The best cost rate for Machine {machine_name} is:", round(cm_cost_rate, 2))
    return


def run_analysis():
    # Loop to iterate through the machines
    machine_names = [1, 2, 3]

    for machine_name in machine_names:
        # Age-based maintenance data loading
        machine_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}.csv')
        cost_data = pd.read_csv(f'{data_path}{student_nr}-Costs.csv').loc[machine_name - 1]
        PM_cost, CM_cost = cost_data.iloc[1], cost_data.iloc[2]

        # Running data preparation for age-based
        prepared_data = data_preparation(machine_data)

        # Kaplan_Meier Estimations
        KM_data = create_kaplanmeier_data(prepared_data)
        MTBF_KM = meantimebetweenfailure_KM(KM_data)

        # Weibull Estimations
        l, k = fit_weibull_distribution(prepared_data)
        MTBF_weibull = meantimebetweenfailure_weibull(l, k)
        weibull_data = create_weibull_curve_data(prepared_data, l, k)

        # Visualize Kaplan-Meier and Weibull
        visualization(KM_data, weibull_data, machine_name)

        # Policy evaluation
        best_age, best_cost_rate = create_cost_data(prepared_data, l, k, PM_cost, CM_cost, machine_name)
        """
        # Uncomment if need check
        print(f'The optimal maintenance age for Machine {machine_name} is', round(best_age, 2))
        print(f'The best cost rate for Machine {machine_name} is', round(best_cost_rate, 2))
        """

        # Condition-based maintenance
        if machine_name == 3:
            condition_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}-condition-data.csv')
            prepared_condition_data = CBM_data_preparation(condition_data)
            failure_level = int(prepared_condition_data['Condition'].max())
            CBM_threshold, CBM_cost_rate = CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name)
            """
            # Uncomment if need check
            print(f'The optimal cost rate under CBM for Machine {machine_name} is: ', round(CBM_cost_rate, 2))
            print(f'The optimal CBM threshold for Machine {machine_name} is: ', CBM_threshold)
            """

        # Purely corrective maintenance cost rate
        cm_cost_rate = cm_cost_rate_calculation(MTBF_KM, MTBF_weibull, CM_cost)

        # Report on results
        print(f"MACHINE {machine_name}", )
        print(f"The MTBF-KaplanMeier for Machine {machine_name}  is: ", round(MTBF_KM, 2))
        print(f"The MTBF-Weibull for Machine {machine_name} is: ", round(MTBF_weibull, 2))

        # Logic to make sure only possible maintenance policies are compared
        # Warning: All savings calculated in time unit, according to the answers in Q&A
        if machine_name == 1 or machine_name == 2:
            # Check if TBM is optimal
            if best_cost_rate < cm_cost_rate:
                report_tbm(machine_name, best_cost_rate, cm_cost_rate)
            # Check if CM is optimal
            elif cm_cost_rate < best_cost_rate:
                report_cm(machine_name, cm_cost_rate)

        if machine_name == 3:
            # Check if TBM is optimal
            if best_cost_rate < cm_cost_rate and best_cost_rate < CBM_cost_rate:
                report_tbm(machine_name, best_cost_rate, cm_cost_rate)
            # Check if CM is optimal
            elif cm_cost_rate < best_cost_rate and cm_cost_rate < CBM_cost_rate:
                report_cm(machine_name, cm_cost_rate)
            # Check if CBM is optimal
            elif CBM_cost_rate < cm_cost_rate and CBM_cost_rate < best_cost_rate:
                report_cbm(machine_name, CBM_cost_rate, best_cost_rate, cm_cost_rate)

        print("\n")

    return


run_analysis()




