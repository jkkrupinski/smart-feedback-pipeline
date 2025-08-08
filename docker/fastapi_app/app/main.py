from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

# Define the structure of input data
class FeedbackRequest(BaseModel):
    text: str

# Dummy prediction endpoint
@app.post("/predict")
def predict(feedback: FeedbackRequest):
    # For now, we just fake a label
    dummy_label = "feature_request"
    return {"label": dummy_label}