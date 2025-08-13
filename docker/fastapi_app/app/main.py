from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from .utils import clean_text

from pathlib import Path


app = FastAPI()

# Load model
MODEL_PATH = Path("models/model.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError("You must train the model first!")

model = joblib.load(MODEL_PATH)


# Define the structure of input data
class FeedbackRequest(BaseModel):
    text: str


# Prediction endpoint
@app.post("/predict")
def predict(feedback: FeedbackRequest):
    cleaned = clean_text(feedback.text)
    prediction = model.predict([cleaned])[0]
    return {"label": prediction}
