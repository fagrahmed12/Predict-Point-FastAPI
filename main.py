import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Download the model and scaler
model = joblib.load("svm_model.pkl")  
scaler = joblib.load("scaler.pkl")

# FastAPI
app = FastAPI()

# Definition of the expected format of the incoming data
class IVData(BaseModel):
    readings: list  # List containing voltage and current values

# classes
classes = ["aging", "crack", "hotspot", "normal", "pid", "shading", "shortcircuit"]

# Feature extraction function from data
def calc_features(data_array):
    try:
        f1 = data_array.mean(axis=0)
        f2 = data_array.std(axis=0)
        f3 = data_array.min(axis=0)
        f4 = data_array.max(axis=0)
        features = np.concatenate([f1, f2, f3, f4])
        return features
    except Exception:
        return [0] * 18  # في حالة الخطأ

# API Endpoint Definition for receiving data from the application
@app.post("/predict")
async def predict(data: IVData):
    # Convert list to numpy array
    iv_array = np.array(data.readings).reshape(-1, 2)

    # Feature extraction
    new_features = calc_features(iv_array).reshape(1, -1)

    #  StandardScaler
    new_features_scaled = scaler.transform(new_features)

    # Predict the class using the model
    prediction = model.predict(new_features_scaled)

    # Returns defect type based on number
    return {"predicted_class": classes[prediction[0]]}

#  Run the application when the file is played directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # استخدام المنفذ من متغير البيئة
    uvicorn.run(app, host="0.0.0.0", port=port)
