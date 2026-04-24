import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# قراءة البيانات
df = pd.read_csv("beauty_ai_dataset.csv")

# تحويل النصوص لأرقام
le_skin = LabelEncoder()
le_symptoms = LabelEncoder()
le_result = LabelEncoder()

df["skin_enc"] = le_skin.fit_transform(df["skin_type"])
df["symptoms_enc"] = le_symptoms.fit_transform(df["symptoms"])
df["result_enc"] = le_result.fit_transform(df["recommendation"])

# المدخلات والمخرجات
X = df[["skin_enc", "symptoms_enc"]]
y = df["result_enc"]

# تدريب النموذج
model = RandomForestClassifier()
model.fit(X, y)

# حفظ النموذج
joblib.dump(model, "model.pkl")
joblib.dump(le_skin, "skin_encoder.pkl")
joblib.dump(le_symptoms, "symptoms_encoder.pkl")
joblib.dump(le_result, "result_encoder.pkl")

print("Model trained successfully ✅")