import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


from utils import clean_text


def train_model():
    # Load data
    df = pd.read_csv("data/raw/feedback.csv")
    
    X = df["text"].apply(clean_text)
    y = df["label"]
    
    # Build pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])
    
    # Train model
    pipeline.fit(X, y)

    # Evaluation
    preds = pipeline.predict(X)
    print("Classification Report:")
    print(classification_report(y, preds))


    # Save model
    joblib.dump(pipeline, "models/model.pkl")
    print("Model trained and saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
