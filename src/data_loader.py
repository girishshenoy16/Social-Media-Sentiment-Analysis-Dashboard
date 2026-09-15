"""Data integrity validator and TweetEval dataset parser."""

import os
from pathlib import Path

REQUIRED_FILES = [
    "mapping.txt",
    "train_text.txt",
    "train_labels.txt",
    "val_text.txt",
    "val_labels.txt",
    "test_text.txt",
    "test_labels.txt",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def validate_raw_files_exist():
    """Validate all 7 required TweetEval files exist in data/raw/."""
    missing = []
    for fname in REQUIRED_FILES:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"Missing required data files in {RAW_DATA_DIR}: {missing}"
        )
    return True


def parse_mapping():
    """Parse mapping.txt into class_id -> string_label mapping."""
    mapping_path = RAW_DATA_DIR / "mapping.txt"
    class_map = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Malformed mapping line: {line!r}")
            class_id = int(parts[0])
            label = parts[1]
            class_map[class_id] = label
    if not class_map:
        raise ValueError("mapping.txt is empty or contains no valid entries")
    return class_map


def _read_lines(filepath):
    """Read lines from a file, stripping trailing newlines."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.rstrip("\n").rstrip("\r") for line in f]


def _validate_split(text_file, label_file, split_name):
    """Validate that text and label files for a split have matching line counts."""
    texts = _read_lines(text_file)
    labels = _read_lines(label_file)
    if len(texts) != len(labels):
        raise ValueError(
            f"Line count mismatch for {split_name}: "
            f"{len(texts)} texts vs {len(labels)} labels"
        )
    return texts, labels


def load_dataset():
    """Load and validate the complete TweetEval dataset.

    Returns:
        dict with keys: class_map, train_texts, train_labels,
        val_texts, val_labels, test_texts, test_labels, stats
    """
    validate_raw_files_exist()
    class_map = parse_mapping()

    train_texts, train_labels = _validate_split(
        RAW_DATA_DIR / "train_text.txt",
        RAW_DATA_DIR / "train_labels.txt",
        "train",
    )
    val_texts, val_labels = _validate_split(
        RAW_DATA_DIR / "val_text.txt",
        RAW_DATA_DIR / "val_labels.txt",
        "validation",
    )
    test_texts, test_labels = _validate_split(
        RAW_DATA_DIR / "test_text.txt",
        RAW_DATA_DIR / "test_labels.txt",
        "test",
    )

    # Validate all labels are in mapping
    all_labels = set(train_labels + val_labels + test_labels)
    valid_label_ids = set(class_map.keys())
    invalid = all_labels - {str(lid) for lid in valid_label_ids}
    if invalid:
        raise ValueError(
            f"Found labels not in mapping.txt: {invalid}"
        )

    # Convert labels to integers
    train_labels = [int(l) for l in train_labels]
    val_labels = [int(l) for l in val_labels]
    test_labels = [int(l) for l in test_labels]

    stats = {
        "train_count": len(train_texts),
        "val_count": len(val_texts),
        "test_count": len(test_texts),
        "total_count": len(train_texts) + len(val_texts) + len(test_texts),
        "num_classes": len(class_map),
        "class_map": {str(k): v for k, v in class_map.items()},
    }

    return {
        "class_map": class_map,
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "stats": stats,
    }


def save_processed_data(train_texts, train_labels, val_texts, val_labels,
                        test_texts, test_labels):
    """Save processed data as CSV files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for texts, labels, fname in [
        (train_texts, train_labels, "train_cleaned.csv"),
        (val_texts, val_labels, "val_cleaned.csv"),
        (test_texts, test_labels, "test_cleaned.csv"),
    ]:
        with open(PROCESSED_DATA_DIR / fname, "w", encoding="utf-8") as f:
            f.write("text,label\n")
            for text, label in zip(texts, labels):
                escaped = text.replace('"', '""')
                f.write(f'"{escaped}",{label}\n')
