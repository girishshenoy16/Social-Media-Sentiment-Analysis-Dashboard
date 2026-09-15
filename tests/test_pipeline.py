"""Tests for the ML pipeline - preprocessing, training, evaluation."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import clean_text, clean_texts
from src.vectorizer import create_vectorizer, fit_vectorizer, transform_texts
from src.train import train_logistic_regression, train_linear_svm
from src.evaluate import compute_metrics
from src.exporter import validate_export

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_clean_text_lowercasing():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_url_removal():
    assert clean_text("Visit https://example.com now") == "visit now"
    assert clean_text("Check www.google.com") == "check"


def test_clean_text_handle_removal():
    assert clean_text("@user Hello") == "hello"
    assert clean_text("Thanks @john_doe!") == "thanks"


def test_clean_text_special_chars():
    assert clean_text("hello! world#1") == "hello world"


def test_clean_text_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_vectorizer_no_data_leakage():
    """TF-IDF vectorizer must be fit ONLY on training data."""
    from src.data_loader import load_dataset
    dataset = load_dataset()
    train_texts = clean_texts(dataset["train_texts"])

    vectorizer = create_vectorizer()
    X_train = fit_vectorizer(vectorizer, train_texts)

    assert X_train.shape[0] == dataset["stats"]["train_count"]
    assert X_train.shape[1] == 5000


def test_model_training_deterministic():
    """Models must train deterministically with random_state=42."""
    from src.data_loader import load_dataset
    dataset = load_dataset()
    train_texts = clean_texts(dataset["train_texts"])

    vectorizer = create_vectorizer()
    X_train = fit_vectorizer(vectorizer, train_texts)

    lr1 = train_logistic_regression(X_train, dataset["train_labels"])
    lr2 = train_logistic_regression(X_train, dataset["train_labels"])

    assert (lr1.coef_ == lr2.coef_).all()
    assert (lr1.intercept_ == lr2.intercept_).all()


def test_metrics_computation():
    """Metrics must compute correctly."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 1]
    class_map = {0: "negative", 1: "neutral", 2: "positive"}
    metrics = compute_metrics(y_true, y_pred, class_map)
    assert "accuracy" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_web_model_json_exists():
    """web_model.json must exist and be valid JSON."""
    path = PROJECT_ROOT / "docs" / "data" / "web_model.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r") as f:
        data = json.load(f)
    assert "vocabulary" in data
    assert "coefficients" in data


def test_dashboard_data_json_exists():
    """dashboard_data.json must exist and be valid JSON."""
    path = PROJECT_ROOT / "docs" / "data" / "dashboard_data.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r") as f:
        data = json.load(f)
    assert "dataset_stats" in data
    assert "metrics" in data


def test_export_validation():
    """Post-export validation must pass."""
    from src.data_loader import parse_mapping
    class_map = parse_mapping()
    assert validate_export(class_map) is True
