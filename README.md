
# 🚦 Road Accidents Prediction API

This is a FastAPI-based machine learning API that predicts the class of a casualty (Driver, Passenger, or Pedestrian) based on accident-related features.

## 📌 Features
- Predicts casualty class based on various accident parameters.
- Uses **Label Encoding** for categorical values.
- Applies **PowerTransformer** and **StandardScaler** for feature scaling.
- Supports **FastAPI** for fast and efficient API responses.

## 🚀 Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/road-accident-prediction.git
cd road-accident-prediction





pip install -r requirements.txt



uvicorn main:app --reload



Request Body Example:
{
    "Number_of_Vehicles": 2,
    "Time_24hr": 14,
    "First_Road_Class": "A",
    "Road_Surface": "Dry",
    "Lighting_Conditions": "Daylight: street lights present",
    "Weather_Conditions": "Fine without high winds",
    "Casualty_Severity": "Slight",
    "Sex_of_Casualty": "Male",
    "Age_of_Casualty": 30,
    "Type_of_Vehicle": "Car"
}




Response Example:
{
  "prediction_numeric": 0,
  "prediction_text": "Driver"
}


├── api/
│   ├── model.pkl            # Trained machine learning model
│   ├── power_transformer.pkl # PowerTransformer for scaling
│   ├── scaler.pkl            # StandardScaler for scaling
│   ├── label_encoders.pkl    # Encoders for categorical variables
├── main.py                  # FastAPI application
├── requirements.txt          # Required dependencies
├── README.md                 # Project documentation
