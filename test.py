import joblib
import pandas as pd

# تحميل الملفات
model = joblib.load("model.pkl")
skin_enc = joblib.load("skin_encoder.pkl")
symptoms_enc = joblib.load("symptoms_encoder.pkl")
result_enc = joblib.load("result_encoder.pkl")

# تجربة إدخال
skin = "oily"
symptom = "acne and oil"

# تحويل النص إلى أرقام
skin_input = skin_enc.transform([skin])[0]
symptom_input = symptoms_enc.transform([symptom])[0]

# تجهيز الإدخال بشكل مطابق للتدريب
input_data = pd.DataFrame([{
    "skin_enc": int(skin_input),
    "symptoms_enc": int(symptom_input)
}])

# توقع النتيجة
pred = model.predict(input_data)
result = result_enc.inverse_transform(pred)

print("✨ Result:", result[0])