"""TF-IDF feature extraction pipeline."""

import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def create_vectorizer():
    """Create a TfidfVectorizer with the specified configuration.

    Note: random_state is NOT passed as TfidfVectorizer does not expose it.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
    )


def fit_vectorizer(vectorizer, train_texts):
    """Fit the vectorizer ONLY on training text."""
    X_train = vectorizer.fit_transform(train_texts)
    return X_train


def transform_texts(vectorizer, texts):
    """Transform texts using the fitted vectorizer."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path=None):
    """Save the fitted vectorizer to disk."""
    if path is None:
        path = MODELS_DIR / "tfidf_vectorizer.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path=None):
    """Load a fitted vectorizer from disk."""
    if path is None:
        path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
