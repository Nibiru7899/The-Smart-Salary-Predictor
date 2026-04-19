from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('salary_predictor_model.pkl')

@app.get("/")
def home():
    return {"message": "Welcome to the Smart Salary Predictor API!"}

@app.get("/predict")
def predict(years_experience: float):
    input_df = pd.DataFrame([years_experience], columns=['YearsExperience'])

    predicted_salary = model.predict(input_df)
    return {"predicted_salary": predicted_salary[0]}
