"""Tests for data_loader.py - Data integrity validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import validate_raw_files_exist, parse_mapping, load_dataset


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_validate_raw_files_exist():
    """All 7 required TweetEval files must exist."""
    assert validate_raw_files_exist() is True


def test_parse_mapping():
    """mapping.txt must parse to a valid class_map."""
    class_map = parse_mapping()
    assert 0 in class_map
    assert 1 in class_map
    assert 2 in class_map
    assert class_map[0] == "negative"
    assert class_map[1] == "neutral"
    assert class_map[2] == "positive"


def test_load_dataset_line_counts():
    """Train, validation, and test text/label line counts must match."""
    dataset = load_dataset()
    assert dataset["stats"]["train_count"] == len(dataset["train_texts"])
    assert dataset["stats"]["val_count"] == len(dataset["val_texts"])
    assert dataset["stats"]["test_count"] == len(dataset["test_texts"])
    assert len(dataset["train_texts"]) == len(dataset["train_labels"])
    assert len(dataset["val_texts"]) == len(dataset["val_labels"])
    assert len(dataset["test_texts"]) == len(dataset["test_labels"])


def test_load_dataset_total_count():
    """Total count must equal sum of splits."""
    dataset = load_dataset()
    expected = dataset["stats"]["train_count"] + dataset["stats"]["val_count"] + dataset["stats"]["test_count"]
    assert dataset["stats"]["total_count"] == expected


def test_load_dataset_num_classes():
    """Must have exactly 3 classes."""
    dataset = load_dataset()
    assert dataset["stats"]["num_classes"] == 3


def test_all_labels_in_mapping():
    """All label values must exist in mapping.txt."""
    dataset = load_dataset()
    valid_ids = set(dataset["class_map"].keys())
    for split_labels in [dataset["train_labels"], dataset["val_labels"], dataset["test_labels"]]:
        for label in split_labels:
            assert label in valid_ids, f"Label {label} not in mapping"
