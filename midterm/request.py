import requests

# Define the URL of the FastAPI endpoint
url = "http://localhost:9696/predict"

# Define the sample input data
sample_data = [
    {
        "gender": "Female",
        "age": 80,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 30.31,
        "hba1c_level": 6.5,
        "blood_glucose_level": 200
    },
    {
        "gender": "Male",
        "age": 55,
        "hypertension": 1,
        "heart_disease": 1,
        "smoking_history": "former",
        "bmi": 28.5,
        "hba1c_level": 7.0,
        "blood_glucose_level": 180
    }
]

# Make the POST request
response = requests.post(url, json=sample_data)

# Print the response
if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print("Error:", response.status_code, response.text)