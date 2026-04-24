from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 🧠 Lazy Loading (مهم جدًا لـ Render)
model = None
skin_enc = None
symptoms_enc = None
result_enc = None

def load_models():
    global model, skin_enc, symptoms_enc, result_enc

    if model is None:
        model = joblib.load("model.pkl")
        skin_enc = joblib.load("skin_encoder.pkl")
        symptoms_enc = joblib.load("symptoms_encoder.pkl")
        result_enc = joblib.load("result_encoder.pkl")


@app.route("/")
def home():
    return "AI API is running ✅"


@app.route("/predict", methods=["POST"])
def predict():
    load_models()  # 👈 مهم جدًا

    data = request.json

    skin = data["skin"]
    symptom = data["symptom"]

    # تحويل
    skin_input = skin_enc.transform([skin])[0]
    symptom_input = symptoms_enc.transform([symptom])[0]

    input_data = pd.DataFrame([{
        "skin_enc": int(skin_input),
        "symptoms_enc": int(symptom_input)
    }])

    pred = model.predict(input_data)
    result = result_enc.inverse_transform(pred)[0]

    return jsonify({
        "result": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)