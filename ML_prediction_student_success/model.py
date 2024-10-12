import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.utils import resample
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# DATA BALANCING CODE
# Determine the size of the smallest group
min_size = df['Target'].value_counts().min()
seed = 123

# Sample each group to the size of the smallest group
df_graduate = df[df['Target'] == 'Graduate'].sample(min_size, random_state=seed)
df_enrolled = df[df['Target'] == 'Enrolled'].sample(min_size, random_state=seed)
df_dropout = df[df['Target'] == 'Dropout'].sample(min_size, random_state=seed)

# Combine the balanced groups into a new DataFrame
df_balanced = pd.concat([df_graduate, df_enrolled, df_dropout])

y = df['Target']
x = df.drop(columns=['Target'])

# SMOTE for oversampling
sm = SMOTE(random_state=654321)
x_res, y_res = sm.fit_resample(x, y)

# DECISION TREE AND BAGGING CODE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    neg_pred_value = TN / (TN + FN)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    prevalence = (TP + FN) / y_true.size
    detection_rate = TP / y_true.size
    detection_prevalence = (TP + FP) / y_true.size
    balanced_accuracy = (sensitivity + specificity) / 2
    return {
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'Pos Pred Value (Precision)': precision,
        'Neg Pred Value': neg_pred_value,
        'F1 Score': f1,
        'Prevalence': prevalence,
        'Detection Rate': detection_rate,
        'Detection Prevalence': detection_prevalence,
        'Balanced Accuracy': balanced_accuracy
    }

# Evaluate models using cross-validation and calculate metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    kappa_scorer = make_scorer(cohen_kappa_score)
    kappa_scores = cross_val_score(model, X_train, y_train, cv=10, scoring=kappa_scorer)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    return metrics, accuracy_scores.mean(), accuracy_scores.std(), kappa_scores.mean(), kappa_scores.std()

# RANDOM FOREST MODEL CODE
# Load data and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions and evaluate model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# Hyperparameter tuning
param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=10, random_state=42)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Calculate and print model performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
kappa = cohen_kappa_score(y_test, y_pred)
print("Tuned Model Accuracy:", accuracy)
print("Tuned Model Precision:", precision)
print("Tuned Model Recall:", recall)
print("Tuned Model F1 Score:", f1)
print("Tuned Model Cohen's kappa:", kappa)

# Instantiate and fit RandomForestClassifier with OOB score
rf = RandomForestClassifier(oob_score=True, random_state=42)
rf.fit(X_train, y_train)
oob_score = rf.oob_score_
print("OOB score:", oob_score)
