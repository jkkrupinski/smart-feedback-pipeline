from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

from .utils import clean_text

from pathlib import Path


app = FastAPI()

USE_DUMMY = os.getenv("USE_DUMMY_MODEL", "false").lower() == "true"

model_path = "models/model_dummy.pkl" if USE_DUMMY else "models/model.pkl"

MODEL_PATH = Path(model_path)
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
