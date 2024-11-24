import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction import DictVectorizer

# Load the model
model_file = 'best_random_forest_model.joblib'
model = joblib.load(model_file)

# Load the DictVectorizer
dv_file = 'dict_vectorizer.joblib'
dv = joblib.load(dv_file)

# Define the app
app = FastAPI()

# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the prediction request body
class InputFeatures(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hba1c_level: float
    blood_glucose_level: float

# Function to set age range and BMI group
def set_age_range_and_bmi_group(user_params):
    # Define bins and labels for age and BMI
    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    bmi_bins = [0, 18.5, 24.9, 29.9, 100]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
    
    # Determine the age range
    user_age = user_params['age']
    user_params['age_range'] = pd.cut([user_age], bins=age_bins, labels=age_labels)[0]
    
    # Determine the BMI group
    user_bmi = user_params['bmi']
    user_params['bmi_group'] = pd.cut([user_bmi], bins=bmi_bins, labels=bmi_labels)[0]
    
    return user_params

# Function to create DataFrame from received data
def create_dataframe(data: List[InputFeatures]) -> pd.DataFrame:
    # Convert input data to DataFrame
    df = pd.DataFrame([item.dict() for item in data])
    
    # Apply age range and BMI group
    df = df.apply(set_age_range_and_bmi_group, axis=1)
    
    X_encoded = dv.transform(df.to_dict(orient='records'))
    
    # Combine numeric and categorical features
    # X_numeric = df[numeric_features].reset_index(drop=True)
    X_processed = pd.concat([pd.DataFrame(X_encoded)], axis=1)
    
    return X_processed

# Define the prediction endpoint
@app.post("/predict")
def predict(data: List[InputFeatures]):
    df = create_dataframe(data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}