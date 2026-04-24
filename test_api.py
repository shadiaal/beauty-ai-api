import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "skin": "oily",
    "symptom": "acne and oil"
}

response = requests.post(url, json=data)

print(response.json())