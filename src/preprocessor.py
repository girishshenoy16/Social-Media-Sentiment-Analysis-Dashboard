"""Deterministic text cleaning engine for sentiment analysis."""

import re


def clean_text(text):
    """Apply deterministic text cleaning sequence.

    Contract (must match JavaScript predictor.js exactly):
    1. Lowercasing
    2. URL removal
    3. User handle removal
    4. Special character cleaning (keep only alphabetic and spaces)
    5. Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercasing
    text = text.lower()

    # 2. URL removal
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. User handle removal
    text = re.sub(r'@\w+', '', text)

    # 4. Special character cleaning - keep only alphabetic and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_texts(texts):
    """Clean a list of texts."""
    return [clean_text(t) for t in texts]
