import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def train_model():
    # Load data
    df = pd.read_csv("data/raw/feedback.csv")
    
    X = df["text"]
    y = df["label"]
    
    # Build pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])
    
    # Train model
    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, "models/model.pkl")
    print("Model trained and saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
