import spacy
import string

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:

    text = text.lower().strip()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Process with spaCy
    doc = nlp(text)

    # Lemmatize and remove stopwords
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return " ".join(tokens)
