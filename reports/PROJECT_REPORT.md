# Project Report — Social Media Sentiment Analysis Dashboard

## 1. Project Overview

This project implements a complete sentiment analysis system for social media text, from data ingestion through model training to a production-ready web dashboard. The system classifies tweets into three sentiment categories: negative, neutral, and positive.

## 2. Dataset

### TweetEval Sentiment Benchmark

The project uses the TweetEval benchmark dataset, a widely-cited evaluation framework for tweet classification:

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 45,615 | 76.2% |
| Validation | 2,000 | 3.3% |
| Test | 12,284 | 20.5% |
| **Total** | **59,899** | **100%** |

**Class Distribution (combined dataset):**

| Class | Count | Percentage |
|-------|-------|------------|
| Negative | 11,377 | 19.0% |
| Neutral | 27,479 | 45.9% |
| Positive | 21,043 | 35.1% |

The neutral class dominates, which is characteristic of social media text where many tweets express factual observations without strong sentiment.

### Test-Set Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Negative | 3,972 | 32.3% |
| Neutral | 5,937 | 48.3% |
| Positive | 2,375 | 19.3% |

## 3. Methodology

### 3.1 Data Integrity

All 7 raw TweetEval files were validated for existence, matching line counts between text and label files, and consistency with the mapping file. The dataset was loaded without modification — no re-shuffling or re-splitting occurred.

### 3.2 Preprocessing

A deterministic 5-step cleaning pipeline was applied:

1. **Lowercasing** — All text converted to lowercase
2. **URL removal** — HTTP/HTTPS URLs and www links removed
3. **Handle removal** — @username mentions removed
4. **Special character cleaning** — Only alphabetic characters and spaces retained
5. **Whitespace normalization** — Multiple spaces collapsed to single space

This preprocessing contract is implemented identically in both Python (`src/preprocessor.py`) and JavaScript (`docs/js/predictor-core.js`).

### 3.3 Feature Engineering

- **TF-IDF Vectorization** with `TfidfVectorizer`:
  - N-gram range: (1, 2) — unigrams and bigrams
  - Max features: 5,000
  - Sublinear TF scaling enabled
  - L2 normalization
  - Smooth IDF

**Critical data integrity constraint:** The vectorizer was fit **only on training text**. Validation and test sets were transformed using the training-fitted vectorizer, preventing any data leakage.

### 3.4 Model Training

**Logistic Regression (Primary Model):**
- Solver: lbfgs
- `random_state=42` for reproducibility
- Max iterations: 1,000

**Linear SVM (Benchmark Model):**
- `random_state=42` for reproducibility
- Max iterations: 2,000

Both models were trained **only on training data** using the TF-IDF features.

## 4. Results

### 4.1 Test Set Performance

| Metric | Logistic Regression | Linear SVM |
|--------|-------------------|------------|
| Accuracy | 58.3% | 57.2% |
| Macro Precision | 58.0% | 56.2% |
| Macro Recall | 56.4% | 55.9% |
| Macro F1-Score | 55.8% | 55.3% |

### 4.2 Analysis

Logistic Regression outperforms Linear SVM across all metrics. The performance difference, while modest, is consistent. The models achieve reasonable accuracy given the inherent difficulty of sentiment classification on short, informal text.

Key challenges:
- Class imbalance (neutral dominates)
- Sarcasm and implicit sentiment
- Short text length limiting contextual information
- OOV terms and informal language

### 4.3 Confusion Matrix Analysis

The confusion matrices reveal that the neutral class is most accurately predicted, while negative and positive classes show more confusion with neutral — expected given the boundary cases in natural language.

**Logistic Regression confusion matrix (test set):**

|  | Negative | Neutral | Positive |
|--|----------|---------|----------|
| **Negative** | 1,510 | 2,042 | 420 |
| **Neutral** | 761 | 4,219 | 957 |
| **Positive** | 103 | 842 | 1,430 |

## 5. Web Deployment Architecture

### 5.1 Export Process

The fitted TF-IDF vectorizer parameters and Logistic Regression weights were exported to `docs/data/web_model.json`:
- Vocabulary mapping (5,000 terms)
- IDF values
- Coefficient matrix (3 classes × 5,000 features)
- Intercept vector
- Class mapping
- Top positive/negative terms per class

Dashboard data is exported to `docs/data/dashboard_data.json`:
- Dataset statistics (split sizes, class distribution, test-set class distribution)
- Model metrics for both Logistic Regression and Linear SVM
- Class labels

### 5.2 Browser-Based Inference

The JavaScript prediction engine is split into two modules:

**`docs/js/predictor-core.js`** — Shared prediction core (browser + Node.js compatible):
1. Identical text preprocessing (`cleanText`)
2. N-gram generation (`generateNgrams`)
3. TF-IDF vectorization using exported vocabulary and IDF
4. L2 normalization
5. Matrix multiplication with exported coefficients
6. Softmax probability calculation

**`docs/js/predictor.js`** — Browser-specific wrapper:
- Loads `web_model.json` via `fetch()`
- Delegates prediction to `predictor-core.js`
- Exposes `SentimentPredictor` API for the dashboard

**No network requests are made after initial page load.**

### 5.3 Python ↔ JavaScript Parity

25 canonical test sentences (defined in `tests/parity_test_cases.json`) are used for parity verification:
- **100% class-label agreement** between Python and JavaScript predictions
- **Probability differences = 0.000000** across all classes

The parity validation runs in three contexts:
- `tests/test_parity.py` — pytest (Python predicts, JS predicts via subprocess)
- `docs/js/test-runner.js` — Node.js CLI (JS predicts, compares against Python reference)
- Browser "Run Test Cases" button (JS predicts, compares against Python reference)

Python is the sole reference generator. JavaScript never writes reference files.

### 5.4 Softmax Verification

For this specific `LogisticRegression` configuration (`multi_class='auto'`, `solver='lbfgs'`, 3 classes):
- `softmax(model.decision_function(X))` is mathematically equivalent to `model.predict_proba(X)`
- Maximum difference across all 25 test sentences: **0.00e+00**

## 6. Testing

### 6.1 Test Suite

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_data_loader.py | 6 | Data integrity, mapping, line counts |
| test_pipeline.py | 12 | Preprocessing, vectorization, training, metrics, export |
| test_parity.py | 1 | 25-sentence Python/JS parity |
| **Total** | **19** | **All pass** |

### 6.2 Key Test Results

- All 19 pytest tests pass
- 25/25 parity test sentences achieve exact class agreement
- Maximum probability difference: 0.000000
- Zero data leakage (vectorizer fit only on training data)
- Deterministic training verified (identical results with random_state=42)

## 7. Dashboard

A single-page, 3-section Power BI-inspired dashboard:

1. **Executive Overview** — KPI ribbon (total samples, accuracy, F1-score, classes), sentiment donut chart (data-driven from `dashboard_data.json`), dataset split bar chart, top sentiment-associated terms
2. **Interactive Sentiment Analyzer** — Real-time prediction with confidence score and probability breakdown, automated 25-case parity test runner, lexicon-based demo classifier
3. **Model Performance & Business Insights** — Metrics comparison table, class-wise performance chart, confusion matrices, data-derived business insights

### 7.1 Data Integrity

All dashboard visualizations are driven by Python-generated data:

| Visualization | Data Source | Hardcoded? |
|---------------|-------------|------------|
| KPI: Total Samples | `dashboard_data.json` | No |
| KPI: Accuracy | `dashboard_data.json` | No |
| KPI: Macro F1 | `dashboard_data.json` | No |
| Sentiment Donut | `dashboard_data.json` → `class_distribution` | No |
| Dataset Split | `dashboard_data.json` | No |
| Top Terms | `web_model.json` | No |
| Metrics Table | `dashboard_data.json` | No |
| Class Performance | `dashboard_data.json` | No |
| Confusion Matrices | PNG files (auto-copied by pipeline) | No |
| Business Insights | `dashboard_data.json` | No |
| Negative Alert | `dashboard_data.json` → `test_class_distribution` | No |

### 7.2 Negative Sentiment Alert

The dashboard includes a configurable alert for negative sentiment in the test set:
- **Threshold:** >25%
- **Data source:** `test_class_distribution.negative` from `dashboard_data.json`
- **Actual value:** 3,972 / 12,284 = 32.3%
- **Triggered:** Yes (32.3% > 25%)

## 8. Deployment

The dashboard is served from the `/docs` directory via GitHub Pages:
- Zero runtime dependencies after initial load
- All assets served as static files
- Chart.js loaded from CDN (only external dependency)
- Compatible with GitHub Pages deployment model
- Local verification: `python -m http.server 8000 --directory docs`

## 9. Reproducibility

- `random_state=42` for all stochastic components
- Fixed TF-IDF configuration
- Deterministic preprocessing pipeline
- Dataset splits preserved from TweetEval benchmark
- All dependencies pinned in `requirements.txt`
- Pipeline (`python main.py`) regenerates all artifacts including confusion matrix copies

## 10. Repository Structure

```
Social Media Sentiment Analysis Dashboard/
├── data/
│   ├── raw/                       # Immutable TweetEval Benchmark Dataset
│   └── processed/                 # Processed cleaned datasets
├── src/                           # Modular Python ML & NLP Pipeline
│   ├── data_loader.py             # Data integrity validator & parser
│   ├── preprocessor.py            # Deterministic text cleaning engine
│   ├── vectorizer.py              # TF-IDF feature extraction pipeline
│   ├── train.py                   # Deterministic model trainer
│   ├── evaluate.py                # Test set evaluation & confusion matrix
│   └── exporter.py                # Web artifact exporter
├── models/                        # Serialized Python Model Artifacts (.pkl)
├── outputs/                       # Evaluation Plots & Metrics JSON
├── tests/                         # Pytest Test Suite
│   ├── test_data_loader.py
│   ├── test_pipeline.py
│   ├── test_parity.py             # Python ↔ JS parity test
│   └── parity_test_cases.json     # Canonical 25 test sentences
├── docs/                          # GitHub Pages Root Web Directory
│   ├── index.html
│   ├── css/style.css
│   ├── js/
│   │   ├── predictor-core.js      # Shared prediction core (browser+Node)
│   │   ├── predictor.js           # Browser prediction API
│   │   ├── dashboard.js           # Chart.js visualizations
│   │   ├── app.js                 # Navigation & UI controller
│   │   └── test-runner.js         # Node.js parity CLI
│   ├── data/
│   │   ├── web_model.json         # Exported LR weights & TF-IDF config
│   │   ├── dashboard_data.json    # Dashboard statistics & metrics
│   │   └── parity_reference.json  # Python reference for browser tests
│   └── outputs/                   # Confusion matrix images (auto-copied)
├── reports/                       # Documentation Reports
├── main.py                        # Pipeline CLI entrypoint
├── requirements.txt
└── README.md
```
