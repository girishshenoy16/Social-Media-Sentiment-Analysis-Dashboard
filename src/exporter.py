"""Export web artifacts for browser-based inference."""

import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"

_TEST_SENTENCES_PATH = PROJECT_ROOT / "tests" / "parity_test_cases.json"
TEST_SENTENCES = json.loads(_TEST_SENTENCES_PATH.read_text(encoding="utf-8"))


def get_top_terms(vectorizer, model, class_labels, class_to_coef_index,
                  top_n=15):
    """Extract top positive and negative terms per class."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_

    results = {}
    for class_idx, class_label in class_labels.items():
        coef_idx = class_to_coef_index[class_idx]
        coef = coefs[coef_idx]

        top_positive_idx = np.argsort(coef)[-top_n:][::-1]
        top_negative_idx = np.argsort(coef)[:top_n]

        results[class_label] = {
            "positive_terms": [feature_names[i] for i in top_positive_idx],
            "negative_terms": [feature_names[i] for i in top_negative_idx],
        }
    return results


def export_web_model(vectorizer, model, class_map, model_name="logistic_regression"):
    """Export fitted vectorizer and LR model to web_model.json."""
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    feature_names = list(vectorizer.get_feature_names_out())
    idf = vectorizer.idf_.tolist()

    ngram_range = vectorizer.ngram_range
    max_features = vectorizer.max_features
    sublinear_tf = vectorizer.sublinear_tf
    norm = vectorizer.norm
    use_idf = vectorizer.use_idf
    smooth_idf = vectorizer.smooth_idf

    coefficients = model.coef_.tolist()
    intercept = model.intercept_.tolist()
    classes = [int(c) for c in model.classes_]

    class_labels = {str(k): v for k, v in class_map.items()}
    class_to_coef_index = {str(k): i for i, k in enumerate(sorted(class_map.keys()))}

    top_terms = get_top_terms(vectorizer, model, class_labels, class_to_coef_index)

    web_model = {
        "vocabulary": vocabulary,
        "feature_names": feature_names,
        "idf": idf,
        "ngram_range": list(ngram_range),
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
        "norm": norm,
        "use_idf": use_idf,
        "smooth_idf": smooth_idf,
        "coefficients": coefficients,
        "intercept": intercept,
        "classes": classes,
        "class_labels": class_labels,
        "class_to_coef_index": class_to_coef_index,
        "top_terms": top_terms,
    }

    output_path = DOCS_DATA_DIR / "web_model.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(web_model, f, cls=NumpyEncoder)

    return output_path


def export_dashboard_data(dataset_stats, lr_metrics, svm_metrics, class_map,
                         all_labels=None, test_labels=None):
    """Export dashboard data including stats, metrics, and insights.

    Args:
        dataset_stats: Dict with train_count, val_count, test_count, total_count, etc.
        lr_metrics: Logistic Regression metrics dict.
        svm_metrics: Linear SVM metrics dict.
        class_map: Dict mapping class_id (int) to string label.
        all_labels: Optional list of all labels (train+val+test) for combined class distribution.
        test_labels: Optional list of test-set labels for test class distribution.
    """
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    class_labels = {str(k): v for k, v in class_map.items()}

    dashboard_data = {
        "dataset_stats": dataset_stats,
        "metrics": {
            "logistic_regression": lr_metrics,
            "linear_svm": svm_metrics,
        },
        "class_labels": class_labels,
    }

    if all_labels is not None:
        combined_dist = {}
        for class_id in sorted(class_map.keys()):
            label = class_map[class_id]
            combined_dist[label] = sum(1 for l in all_labels if l == class_id)
        dashboard_data["dataset_stats"]["class_distribution"] = combined_dist

    if test_labels is not None:
        test_dist = {}
        for class_id in sorted(class_map.keys()):
            label = class_map[class_id]
            test_dist[label] = sum(1 for l in test_labels if l == class_id)
        dashboard_data["dataset_stats"]["test_class_distribution"] = test_dist

    output_path = DOCS_DATA_DIR / "dashboard_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)

    return output_path


def export_parity_reference(vectorizer, model, class_map):
    """Generate parity reference JSON from Python TF-IDF + LR model.

    Writes to both:
      - tests/python_reference.json  (canonical Python reference)
      - docs/data/parity_reference.json  (browser-accessible copy)
    """
    from src.preprocessor import clean_text

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    class_labels = {str(k): v for k, v in class_map.items()}

    refs = []
    for sent in TEST_SENTENCES:
        cleaned = clean_text(sent)
        X = vectorizer.transform([cleaned])
        decision = model.decision_function(X)[0]
        exp_vals = np.exp(decision - np.max(decision))
        probs = exp_vals / np.sum(exp_vals)
        pred_idx = int(model.predict(X)[0])
        probs_dict = {class_labels[str(i)]: float(probs[i]) for i in range(len(probs))}
        refs.append({"class_label": class_labels[str(pred_idx)], "probabilities": probs_dict})

    tests_path = PROJECT_ROOT / "tests" / "python_reference.json"
    with open(tests_path, "w", encoding="utf-8") as f:
        json.dump(refs, f, indent=2)

    docs_path = DOCS_DATA_DIR / "parity_reference.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(refs, f, indent=2)

    print(f"Parity reference exported: {tests_path} and {docs_path} ({len(refs)} sentences)")
    return tests_path


def validate_export(class_map):
    """Post-export validation checks."""
    model_path = DOCS_DATA_DIR / "web_model.json"
    dash_path = DOCS_DATA_DIR / "dashboard_data.json"

    assert model_path.exists(), f"Missing {model_path}"
    assert dash_path.exists(), f"Missing {dash_path}"

    with open(model_path, "r") as f:
        model = json.load(f)
    with open(dash_path, "r") as f:
        dash = json.load(f)

    assert len(model["vocabulary"]) == len(model["feature_names"]), \
        "Vocabulary and feature_names length mismatch"
    assert len(model["vocabulary"]) == len(model["idf"]), \
        "Vocabulary and idf length mismatch"

    n_classes = len(model["classes"])
    n_features = len(model["feature_names"])
    assert len(model["coefficients"]) == n_classes, \
        f"Coefficients rows {len(model['coefficients'])} != num_classes {n_classes}"
    for i, row in enumerate(model["coefficients"]):
        assert len(row) == n_features, \
            f"Coefficients row {i} has {len(row)} features, expected {n_features}"

    assert len(model["intercept"]) == n_classes, \
        f"Intercept length {len(model['intercept'])} != num_classes {n_classes}"

    assert model["class_labels"] == {str(k): v for k, v in class_map.items()}, \
        "Class labels mismatch with mapping.txt"

    print("Post-export validation passed: web_model.json and dashboard_data.json")
    return True
