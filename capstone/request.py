import requests

# Define the URL of the FastAPI endpoint
url = "http://localhost:9696/predict"

# Define the sample input data
sample_data = [
    {
        'age': 54, 
        'sex': 'M', 
        'chestpaintype': 'ASY', 
        'restingbp': 125, 
        'cholesterol': 224, 
        'fastingbs': 0, 
        'restingecg': 'Normal', 
        'maxhr': 122, 
        'exerciseangina': 'N', 
        'oldpeak': 2.0, 
        'st_slope': 'Flat'
    },
    {
        'age': 54, 
        'sex': 'M', 
        'chestpaintype': 'ASY', 
        'restingbp': 125, 
        'cholesterol': 224, 
        'fastingbs': 0, 
        'restingecg': 'Normal', 
        'maxhr': 122, 
        'exerciseangina': 'N', 
        'oldpeak': 2.0, 
        'st_slope': 'Flat'
    }
]

# Make the POST request
response = requests.post(url, json=sample_data)

# Print the response
if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print("Error:", response.status_code, response.text)