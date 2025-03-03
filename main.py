import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# تحميل المودل والـ scaler
model = joblib.load("svm_model.pkl")  # تأكد من رفع هذا الملف إلى Railway
scaler = joblib.load("scaler.pkl")

# تعريف FastAPI
app = FastAPI()

# تعريف الشكل المتوقع للبيانات الواردة
class IVData(BaseModel):
    readings: list  # قائمة تحتوي على قيم الجهد والتيار

# أسماء الـ classes
classes = ["aging", "crack", "hotspot", "normal", "pid", "shading", "shortcircuit"]

# دالة استخراج الميزات من البيانات
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

# تعريف API Endpoint لاستقبال البيانات من التطبيق
@app.post("/predict")
async def predict(data: IVData):
    # تحويل القائمة إلى numpy array
    iv_array = np.array(data.readings).reshape(-1, 2)

    # استخراج الميزات
    new_features = calc_features(iv_array).reshape(1, -1)

    # تطبيق الـ StandardScaler
    new_features_scaled = scaler.transform(new_features)

    # توقع الـ class باستخدام المودل
    prediction = model.predict(new_features_scaled)

    # إرجاع نوع العيب بناءً على الرقم
    return {"predicted_class": classes[prediction[0]]}

# تشغيل التطبيق عند تشغيل الملف مباشرة
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # استخدام المنفذ من متغير البيئة
    uvicorn.run(app, host="0.0.0.0", port=port)
