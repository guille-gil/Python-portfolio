import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
df = pd.read_csv("data.csv", delimiter=";")

# Drop rows with missing values
df.dropna(inplace=True)

# Load the data
df = pd.read_csv("data.csv", delimiter=";")

# Drop rows with missing values
df.dropna(inplace=True)

# Adjust display options to see all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

print(df.describe(include='all'))
print("Shape of the DataFrame before outlier removal:", df.shape)

columns_to_check = [
        'Application Order','Previous qualification (grade)', 'Admission Grade', 'Age at enrollment', 'Curricular units 1st sem (credited)',
        'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (without evaluations)'
    ]


# Function to remove outliers using the Z-score method for specific columns
def remove_outliers_zscore(df, columns, threshold=3):
    for column in columns:
        if column in df.columns:
            mean = df[column].mean()
            std = df[column].std()
            z_scores = (df[column] - mean) / std
            df = df[np.abs(z_scores) < threshold]
    return df

# Remove outliers using the Z-score method for specific columns
df_clean = remove_outliers_zscore(df, columns_to_check)

# Print the shape after outlier removal
print("Shape of the DataFrame after outlier removal:", df_clean.shape)

# Define mappings for categorical columns
application_mode_mapping = {
    1: '1st phase - general contingent',
    2: 'Ordinance No. 612/93',
    5: '1st phase - special contingent (Azores Island)',
    7: 'Holders of other higher courses',
    10: 'Ordinance No. 854-B/99',
    15: 'International student (bachelor)',
    16: '1st phase - special contingent (Madeira Island)',
    17: '2nd phase - general contingent',
    18: '3rd phase - general contingent',
    26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
    27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
    39: 'Over 23 years old',
    42: 'Transfer',
    43: 'Change of course',
    44: 'Technological specialization diploma holders',
    51: 'Change of institution/course',
    53: 'Short cycle diploma holders',
    57: 'Change of institution/course (International)'
}

course_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}

previous_qualification_mapping = {
    1: 'Secondary education',
    2: 'Higher education - bachelor\'s degree',
    3: 'Higher education - degree',
    4: 'Higher education - master\'s',
    5: 'Higher education - doctorate',
    6: 'Frequency of higher education',
    9: '12th year of schooling - not completed',
    10: '11th year of schooling - not completed',
    12: 'Other - 11th year of schooling',
    14: '10th year of schooling',
    15: '10th year of schooling - not completed',
    19: 'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
    38: 'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
    39: 'Technological specialization course',
    40: 'Higher education - degree (1st cycle)',
    42: 'Professional higher technical course',
    43: 'Higher education - master (2nd cycle)'
}

nationality_mapping = {
    1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian', 13: 'Dutch',
    14: 'English', 17: 'Lithuanian', 21: 'Angolan', 22: 'Cape Verdean',
    24: 'Guinean', 25: 'Mozambican', 26: 'Santomean', 32: 'Turkish',
    41: 'Brazilian', 62: 'Romanian', 100: 'Moldova (Republic of)',
    101: 'Mexican', 103: 'Ukrainian', 105: 'Russian', 108: 'Cuban', 109: 'Colombian'
}

mother_qualification_mapping = {
    1: 'Secondary Education - 12th Year of Schooling or Eq.',
    2: 'Higher Education - Bachelor\'s Degree',
    3: 'Higher Education - Degree',
    4: 'Higher Education - Master\'s',
    5: 'Higher Education - Doctorate',
    6: 'Frequency of Higher Education',
    9: '12th Year of Schooling - Not Completed',
    10: '11th Year of Schooling - Not Completed',
    11: '7th Year (Old)',
    12: 'Other - 11th Year of Schooling',
    14: '10th Year of Schooling',
    18: 'General commerce course',
    19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
    22: 'Technical-professional course',
    26: '7th year of schooling',
    27: '2nd cycle of the general high school course',
    29: '9th Year of Schooling - Not Completed',
    30: '8th year of schooling',
    34: 'Unknown',
    35: 'Can\'t read or write',
    36: 'Can read without having a 4th year of schooling',
    37: 'Basic education 1st cycle (4th/5th year) or equiv.',
    38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
    39: 'Technological specialization course',
    40: 'Higher education - degree (1st cycle)',
    41: 'Specialized higher studies course',
    42: 'Professional higher technical course',
    43: 'Higher Education - Master (2nd cycle)',
    44: 'Higher Education - Doctorate (3rd cycle)'
}

mother_occupation_mapping = {
    0: 'Student', 1: 'Legislative Power & Executive Bodies', 2: 'Intellectual and Scientific Activities',
    3: 'Intermediate Level Technicians and Professions', 4: 'Administrative staff',
    5: 'Personal Services, Security and Safety Workers and Sellers',
    6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
    7: 'Skilled Workers in Industry, Construction and Craftsmen',
    8: 'Installation and Machine Operators and Assembly Workers',
    9: 'Unskilled Workers', 10: 'Armed Forces Professions',
    90: 'Other Situation', 99: '(blank)', 122: 'Health professionals',
    123: 'teachers', 125: 'ICT Specialists',
    131: 'Intermediate level science and engineering technicians and professions',
    132: 'Technicians and professionals, of intermediate level of health',
    134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
    141: 'Office workers, secretaries in general and data processing operators',
    143: 'Data, accounting, statistical, financial services and registry-related operators',
    144: 'Other administrative support staff', 151: 'personal service workers',
    152: 'sellers', 153: 'Personal care workers and the like',
    171: 'Skilled construction workers and the like, except electricians',
    173: 'Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',
    175: 'Workers in food processing, woodworking, clothing and other industries and crafts',
    191: 'cleaning workers', 192: 'Unskilled workers in agriculture, animal production, fisheries and forestry',
    193: 'Unskilled workers in extractive industry, construction, manufacturing and transport',
    194: 'Meal preparation assistants'
}

father_occupation_mapping = {
    0: 'Student', 1: 'Legislative Power & Executive Bodies', 2: 'Intellectual and Scientific Activities',
    3: 'Intermediate Level Technicians and Professions', 4: 'Administrative staff',
    5: 'Personal Services, Security and Safety Workers and Sellers',
    6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
    7: 'Skilled Workers in Industry, Construction and Craftsmen',
    8: 'Installation and Machine Operators and Assembly Workers',
    9: 'Unskilled Workers', 10: 'Armed Forces Professions',
    90: 'Other Situation', 99: '(blank)', 101: 'Armed Forces Officers',
    102: 'Armed Forces Sergeants', 103: 'Other Armed Forces personnel',
    112: 'Directors of administrative and commercial services',
    114: 'Hotel, catering, trade and other services directors',
    121: 'Specialists in the physical sciences, mathematics, engineering and related techniques',
    122: 'Health professionals', 123: 'teachers', 124: 'Specialists in finance, accounting, administrative organization, public and commercial relations',
    131: 'Intermediate level science and engineering technicians and professions',
    132: 'Technicians and professionals, of intermediate level of health',
    134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
    135: 'Information and communication technology technicians',
    141: 'Office workers, secretaries in general and data processing operators',
    143: 'Data, accounting, statistical, financial services and registry-related operators',
    144: 'Other administrative support staff', 151: 'personal service workers',
    152: 'sellers', 153: 'Personal care workers and the like',
    154: 'Protection and security services personnel',
    161: 'Market-oriented farmers and skilled agricultural and animal production workers',
    163: 'Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence',
    171: 'Skilled construction workers and the like, except electricians',
    172: 'Skilled workers in metallurgy, metalworking and similar',
    174: 'Skilled workers in electricity and electronics',
    175: 'Workers in food processing, woodworking, clothing and other industries and crafts',
    181: 'Fixed plant and machine operators', 182: 'assembly workers',
    183: 'Vehicle drivers and mobile equipment operators',
    192: 'Unskilled workers in agriculture, animal production, fisheries and forestry',
    193: 'Unskilled workers in extractive industry, construction, manufacturing and transport',
    194: 'Meal preparation assistants', 195: 'Street vendors (except food) and street service providers'
}

# Replace numeric values with their descriptive counterparts
df['Application mode'] = df['Application mode'].map(application_mode_mapping)
df['Course'] = df['Course'].map(course_mapping)
df['Previous qualification'] = df['Previous qualification'].map(previous_qualification_mapping)
df['Nacionality'] = df['Nacionality'].map(nationality_mapping)
df['Mother\'s qualification'] = df['Mother\'s qualification'].map(mother_qualification_mapping)
df['Father\'s qualification'] = df['Father\'s qualification'].map(mother_qualification_mapping)
df['Mother\'s occupation'] = df['Mother\'s occupation'].map(mother_occupation_mapping)
df['Father\'s occupation'] = df['Father\'s occupation'].map(father_occupation_mapping)

# Convert categorical variables into dummy/indicator variables
categorical_columns = [
    'Application mode', 'Course', 'Previous qualification', 'Nacionality', 'Mother\'s qualification',
    'Father\'s qualification','Mother\'s occupation', 'Father\'s occupation'
]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
print("Dummy variables created successfully.")

# Convert boolean columns to integers because some of them were output as false or true
df = df.astype({col: int for col in df.select_dtypes(include=['bool']).columns})

print(df.columns)

# Sample statistics and frequency for the outcome variable
outcome_variable = 'Target'

# Display frequency counts for the outcome variable
print("Frequency counts for the outcome variable:")
print(df_clean[outcome_variable].value_counts())

# Display proportions for the outcome variable
print("\nProportions for the outcome variable:")
print(df_clean[outcome_variable].value_counts(normalize=True))

# Plot the frequency distribution of the outcome variable
plt.figure(figsize=(10, 5))
sns.countplot(x=outcome_variable, data=df_clean)
plt.title('Frequency Distribution of the Outcome Variable')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.show()

df_clean.to_excel("cleaned_data.xlsx", index=False)
