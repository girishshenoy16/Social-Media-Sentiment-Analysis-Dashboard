"""Pipeline CLI entrypoint for Social Media Sentiment Analysis."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_dataset, save_processed_data
from src.preprocessor import clean_texts
from src.vectorizer import create_vectorizer, fit_vectorizer, transform_texts, save_vectorizer
from src.train import train_logistic_regression, train_linear_svm, save_model
from src.evaluate import evaluate_model, save_metrics
from src.exporter import export_web_model, export_dashboard_data, export_parity_reference, validate_export


def main():
    print("=" * 60)
    print("Social Media Sentiment Analysis Pipeline")
    print("=" * 60)

    # Phase 2: Load and validate data
    print("\n[Phase 2] Loading and validating dataset...")
    dataset = load_dataset()
    stats = dataset["stats"]
    print(f"  Train: {stats['train_count']}, Val: {stats['val_count']}, "
          f"Test: {stats['test_count']}, Total: {stats['total_count']}")
    print(f"  Classes: {stats['class_map']}")

    # Phase 3: Preprocess
    print("\n[Phase 3] Cleaning texts...")
    train_texts = clean_texts(dataset["train_texts"])
    val_texts = clean_texts(dataset["val_texts"])
    test_texts = clean_texts(dataset["test_texts"])

    save_processed_data(
        train_texts, dataset["train_labels"],
        val_texts, dataset["val_labels"],
        test_texts, dataset["test_labels"],
    )
    print("  Processed data saved to data/processed/")

    # Phase 4: Vectorize and train
    print("\n[Phase 4] Vectorizing and training models...")
    vectorizer = create_vectorizer()
    X_train = fit_vectorizer(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)
    X_test = transform_texts(vectorizer, test_texts)
    save_vectorizer(vectorizer)
    print(f"  TF-IDF features: {X_train.shape[1]}")

    print("  Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, dataset["train_labels"])
    save_model(lr_model, "logistic_regression_model.pkl")

    print("  Training Linear SVM...")
    svm_model = train_linear_svm(X_train, dataset["train_labels"])
    save_model(svm_model, "linear_svm_model.pkl")

    # Phase 5: Evaluate
    print("\n[Phase 5] Evaluating models on test set...")
    class_map = dataset["class_map"]

    lr_metrics, lr_preds = evaluate_model(
        lr_model, X_test, dataset["test_labels"], class_map, "Logistic Regression"
    )
    svm_metrics, svm_preds = evaluate_model(
        svm_model, X_test, dataset["test_labels"], class_map, "Linear SVM"
    )

    all_metrics = {
        "logistic_regression": lr_metrics,
        "linear_svm": svm_metrics,
    }
    save_metrics(all_metrics)
    print(f"  Metrics saved to docs/outputs/metrics_summary.json")

    # Phase 6: Export web artifacts
    print("\n[Phase 6] Exporting web artifacts...")
    export_web_model(vectorizer, lr_model, class_map, "logistic_regression")
    print("  Exported docs/data/web_model.json")

    export_dashboard_data(
        stats, lr_metrics, svm_metrics, class_map,
        all_labels=dataset["train_labels"] + dataset["val_labels"] + dataset["test_labels"],
        test_labels=dataset["test_labels"],
    )
    print("  Exported docs/data/dashboard_data.json")

    export_parity_reference(vectorizer, lr_model, class_map)

    validate_export(class_map)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    main()
