"""Tests for Python/JavaScript prediction parity."""

import sys
import json
import subprocess
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data_loader import load_dataset
from src.preprocessor import clean_texts
from src.vectorizer import load_vectorizer
from src.train import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

TEST_SENTENCES = json.loads(
    (PROJECT_ROOT / "tests" / "parity_test_cases.json").read_text(encoding="utf-8")
)


def _build_js_script():
    """Build the JS prediction script using shared predictor-core.js."""
    core_path = str(PROJECT_ROOT / "docs" / "js" / "predictor-core.js").replace("\\", "\\\\")
    model_path = str(PROJECT_ROOT / "docs" / "data" / "web_model.json").replace("\\", "\\\\")
    script = (
        "var fs = require('fs');\n"
        "var core = require('" + core_path + "');\n"
        "var modelData = JSON.parse(fs.readFileSync('" + model_path + "', 'utf8'));\n"
        "var text = process.argv[2];\n"
        "var result = core.predict(text, modelData);\n"
        "console.log(JSON.stringify(result));\n"
    )
    return script


JS_SCRIPT_TEMPLATE = None


def python_predict(text, vectorizer, model, class_map):
    """Run Python prediction on a single text."""
    cleaned = clean_texts([text])
    X = vectorizer.transform(cleaned)

    decision = model.decision_function(X)[0]

    exp_vals = np.exp(decision - np.max(decision))
    probs = exp_vals / np.sum(exp_vals)

    pred_class = int(model.predict(X)[0])

    return {
        "predicted_class": pred_class,
        "class_label": class_map[pred_class],
        "probabilities": {
            class_map[i]: float(p) for i, p in enumerate(probs)
        },
    }


def js_predict(text):
    """Run JavaScript prediction on a single text via Node.js."""
    script = _build_js_script()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(script)
        f.flush()
        temp_path = f.name

    try:
        result = subprocess.run(
            ['node', temp_path, text],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise RuntimeError(f"JS error: {result.stderr}")
        output = result.stdout.strip()
        if output:
            return json.loads(output)
        raise RuntimeError("No output from JS prediction")
    finally:
        os.unlink(temp_path)


def test_parity_25_sentences():
    """25 sentences must produce 100% class-label agreement and prob diffs < 1e-3."""
    dataset = load_dataset()
    class_map = dataset["class_map"]

    vectorizer = load_vectorizer()
    model = load_model("logistic_regression_model.pkl")

    assert len(TEST_SENTENCES) == 25, f"Expected 25 sentences, got {len(TEST_SENTENCES)}"

    agreement_count = 0
    max_prob_diff = 0.0
    failures = []

    for i, sentence in enumerate(TEST_SENTENCES):
        py_result = python_predict(sentence, vectorizer, model, class_map)

        try:
            js_result = js_predict(sentence)
        except Exception as e:
            failures.append((i, sentence, str(e)))
            continue

        py_class = py_result["class_label"]
        js_class = js_result.get("class_label", "")

        if py_class == js_class:
            agreement_count += 1
        else:
            failures.append((i, sentence, f"Python={py_class}, JS={js_class}"))

        for label in class_map.values():
            py_prob = py_result["probabilities"].get(label, 0)
            js_prob = js_result.get("probabilities", {}).get(label, 0)
            diff = abs(py_prob - js_prob)
            if diff > max_prob_diff:
                max_prob_diff = diff

    assert agreement_count == 25, (
        f"Class-label agreement: {agreement_count}/25. Failures: {failures}"
    )
    assert max_prob_diff < 1e-3, (
        f"Max probability difference: {max_prob_diff} >= 1e-3"
    )

    print(f"Parity test PASSED: {agreement_count}/25 class agreement, "
          f"max prob diff: {max_prob_diff:.6f}")
