from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI()

model = joblib.load("api/model.pkl")
power_transformer = joblib.load("api/power_transformer.pkl")
scaler = joblib.load("api/scaler.pkl")
label_encoders = joblib.load("api/label_encoders.pkl")  


prediction_mapping = {
    0: "Driver",
    1: "Passenger",
    2: "Pedestrian"
}


vehicle_group_mapping = {
    'Car': 'Car',
    'Taxi/Private hire car': 'Car',
    'Pedal cycle': 'Two-Wheeled Vehicle',
    'Motorcycle over 50cc and up to 125cc': 'Two-Wheeled Vehicle',
    'Motorcycle over 500cc': 'Two-Wheeled Vehicle',
    'M/cycle 50cc and under': 'Two-Wheeled Vehicle',
    'Motorcycle over 125cc and up to 500cc': 'Two-Wheeled Vehicle',
    'Bus or coach (17 or more passenger seats)': 'Public Transportation',
    'Minibus (8 – 16 passenger seats)': 'Public Transportation',
    'Goods vehicle 3.5 tonnes mgw and under': 'Other Vehicle',
    'Goods vehicle 7.5 tonnes mgw and over': 'Other Vehicle',
    'Goods vehicle over 3.5 tonnes and under 7.5 tonnes mgw': 'Other Vehicle',
    'Other Vehicle': 'Other Vehicle',
    'Ridden horse': 'Other Vehicle',
    'Agricultural vehicle (includes diggers etc.)': 'Other Vehicle'
}


def handle_age(age):
    if age < 70:
        return f"{(age // 10) * 10}-{((age // 10) * 10) + 9}"
    else:
        return "+70"


class AccidentFeatures(BaseModel):
    Number_of_Vehicles: int
    Time_24hr: int
    First_Road_Class: str
    Road_Surface: str
    Lighting_Conditions: str
    Weather_Conditions: str
    Casualty_Severity: str
    Sex_of_Casualty: str
    Age_of_Casualty: int
    Type_of_Vehicle: str

@app.get("/")
def home():
    return {"message": "🚦 Road Accidents Prediction API is Running!"}

@app.get("/health")
def health_check():
    return {"status": "API is working properly"}

@app.post("/predict")
def predict(data: AccidentFeatures):
    try:

        age_group = handle_age(data.Age_of_Casualty)
        vehicle_group = vehicle_group_mapping.get(data.Type_of_Vehicle, "Other Vehicle")

        try:
            input_data = [
                data.Number_of_Vehicles,
                data.Time_24hr,
                label_encoders["1st Road Class"]["encoder"].transform([data.First_Road_Class])[0],
                label_encoders["Road Surface"]["encoder"].transform([data.Road_Surface])[0],
                label_encoders["Lighting Conditions"]["encoder"].transform([data.Lighting_Conditions])[0],
                label_encoders["Weather Conditions"]["encoder"].transform([data.Weather_Conditions])[0],
                label_encoders["Casualty Severity"]["encoder"].transform([data.Casualty_Severity])[0],
                label_encoders["Sex of Casualty"]["encoder"].transform([data.Sex_of_Casualty])[0],
                data.Age_of_Casualty,
                label_encoders["Type of Vehicle"]["encoder"].transform([data.Type_of_Vehicle])[0],
                label_encoders["age_group"]["encoder"].transform([age_group])[0],
                label_encoders["vehicle_group"]["encoder"].transform([vehicle_group])[0]
            ]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"❌ قيمة غير صالحة: {e}")

        input_features = np.array(input_data).reshape(1, -1)


        input_features[:, [1]] = power_transformer.transform(input_features[:, [1]])
        input_features[:, [1]] = scaler.transform(input_features[:, [1]])


        prediction_numeric = model.predict(input_features)[0]

        prediction_text = prediction_mapping.get(int(prediction_numeric), "Unknown")

        return {
            "prediction_numeric": int(prediction_numeric),
            "prediction_text": prediction_text  
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
