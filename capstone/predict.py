import os
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load the model
model_directory = 'model'
model_filename = 'best_xgboost_model.joblib'
model_path = os.path.join(model_directory, model_filename)
model = joblib.load(model_path)

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
    age: int
    sex: str
    chestpaintype: str
    restingbp: int
    cholesterol: int
    fastingbs: int
    restingecg: str
    maxhr: int
    exerciseangina: str
    oldpeak: float
    st_slope: str

# Function to create a DataFrame from the input data
def create_dataframe(data: List[InputFeatures]) -> pd.DataFrame:
    df = pd.DataFrame([item.dict() for item in data])
    # Convert categorical features to 'category' dtype
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')
    return df

# Define the prediction endpoint
@app.post("/predict")
def predict(data: List[InputFeatures]):
    df = create_dataframe(data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}