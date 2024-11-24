#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV

# Path to the CSV file
data_file = 'diabetes_prediction_dataset.csv'

# Loading the data
df = pd.read_csv(data_file)
df.columns = df.columns.str.lower().str.replace(' ', '_')

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Define bins for Age and BMI
age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
df['age_range'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

bmi_bins = [0, 18.5, 24.9, 29.9, 100]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
df['bmi_group'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)

# Split the data into training, validation, and test sets
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)  # 0.25 * 0.8 = 0.2

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.diabetes.values
y_train_full = df_train_full.diabetes.values
y_val = df_val.diabetes.values
y_test = df_test.diabetes.values

del df_train['diabetes']
del df_val['diabetes']
del df_test['diabetes']

# Convert training data to dictionary format
train_dict = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Save the best model and DictVectorizer
model_filename = 'best_random_forest_model.joblib'
joblib.dump(grid_search.best_estimator_, model_filename)
dv_filename = 'dict_vectorizer.joblib'
joblib.dump(dv, dv_filename)