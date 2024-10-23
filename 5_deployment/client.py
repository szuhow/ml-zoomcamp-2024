import requests

url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}

resp = requests.post(url, json=client)

print(resp.json())
