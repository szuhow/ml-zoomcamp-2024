#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import joblib
import numpy as np
import logging
import sklearn 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


df = pd.read_csv('heart.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.loc[df['oldpeak'] < 0, 'oldpeak'] = 0
df.loc[df['cholesterol'] == 0, 'cholesterol'] = df['cholesterol'].median()
df.loc[df['maxhr'] == 0, 'maxhr'] = df['maxhr'].median()


X = df.drop(columns='heartdisease')
y = df['heartdisease']


# In[28]:


# Convert categorical features to 'category' dtype
logging.info("Converting categorical features to 'category' dtype...")
for column in X.select_dtypes(include=['object']).columns:
    X[column] = X[column].astype('category')

logging.info("Splitting the dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.7, 1],
    'colsample_bytree': [0.5, 0.7, 1],
    'gamma': [0, 1, 5]
}

logging.info("Starting GridSearchCV for XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, enable_categorical=True)
grid_search_xgb = GridSearchCV(
    xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search_xgb.fit(X_train, y_train)


logging.info("Completed GridSearchCV for XGBoost Classifier.")

# Get the best estimator
xgb_best_model = grid_search_xgb.best_estimator_
logging.info(f"Best XGBoost model: {xgb_best_model}")


logging.info("Saving the best XGBoost model to disk...")
model_directory = 'model'
model_filename = 'best_xgboost_model.joblib'

# Create the directory if it doesn't exist
os.makedirs(model_directory, exist_ok=True)

# Save the model
model_path = os.path.join(model_directory, model_filename)
joblib.dump(grid_search_xgb.best_estimator_, model_path)

print(f'Model saved to {model_path}')