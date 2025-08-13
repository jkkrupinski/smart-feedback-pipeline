import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

texts = [
    "I can't log in to my account",
    "Please add a dark mode feature",
    "This app is amazing!",
    "I found a bug in the signup page"
]
labels = ["bug", "feature_request", "praise", "bug"]

dummy_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

dummy_pipeline.fit(texts, labels)

with open("models/model_dummy.pkl", "wb") as f:
    pickle.dump(dummy_pipeline, f)

print("Dummy model created at models/model_dummy.pkl")
