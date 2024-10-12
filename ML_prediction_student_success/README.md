# Predictive Modeling for Student Academic Success

## Overview
This project aims to enhance the prediction of student academic success at the Polytechnic Institute of Portalegre by analyzing historical data from 2008 to 2019. Utilizing machine learning techniques such as Decision Trees and Random Forests, the study evaluates different predictive models to forecast academic outcomes across three risk levels. The ultimate goal is to implement effective policies that improve student retention and success rates.

## Folder Structure
- **model.py**: Main script for implementing and evaluating predictive models.
- **data_cleaning.py**: Script for preprocessing and cleaning data, ensuring it is ready for analysis.

### Additional Directories
- **data/**: Contains raw datasets of student records.
- **results/**: Stores outputs like model evaluation reports and charts.
- **cleaned_data/**: Holds processed data sets ready for modeling.

## Datasets
The analysis incorporates a comprehensive dataset that tracks several variables, including academic performance, demographic details, and socio-economic backgrounds of students. This data is used to identify patterns and predict student outcomes effectively.

## Methodologies
### Machine Learning Models
- **Decision Trees and Random Forests**: Used for their efficiency in handling categorical data and their capacity to rank the importance of various features.
- **Data Balancing Techniques**: Implements undersampling and oversampling to address class imbalance in the dataset, enhancing model accuracy and fairness.

### Model Validation
- **Cross-Validation**: Employs 10-fold cross-validation to assess model performance and ensure generalizability.
- **Hyperparameter Tuning**: Uses techniques like Randomized Grid Search to optimize model parameters for best performance.

## Key Findings and Recommendations
The project identifies key predictors of academic success and provides insights into the effectiveness of various intervention strategies. Recommendations for policy adjustments are based on quantitative analysis and are aimed at reducing dropout rates and improving academic achievement.

## Tools and Libraries
- **Python**: For data manipulation and running machine learning algorithms.
- **Pandas** and **NumPy**: For data handling and numerical operations.
- **Scikit-learn**: For modeling and validation.
- **Matplotlib**: For generating visualizations of the model results.

## Conclusion
The application of predictive modeling in educational settings can significantly aid in proactive student support and policy development, ultimately leading to improved educational outcomes and reduced dropout rates.
