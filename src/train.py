"""Deterministic model trainer for Logistic Regression and Linear SVM."""

import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with random_state=42."""
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svm(X_train, y_train):
    """Train Linear SVM with random_state=42."""
    model = LinearSVC(
        random_state=42,
        max_iter=2000,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    """Save a trained model to the models/ directory."""
    path = MODELS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(filename):
    """Load a trained model from the models/ directory."""
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)
