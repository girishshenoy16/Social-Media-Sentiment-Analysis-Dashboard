# Executive Summary — Social Media Sentiment Analysis Dashboard

## Business Problem

Organizations monitoring social media need to automatically classify public sentiment toward brands, products, and campaigns. Manual analysis of millions of daily social media posts is infeasible, creating demand for automated sentiment analysis at scale.

## Solution Overview

An end-to-end sentiment analysis system that classifies social media text as **positive**, **neutral**, or **negative**. The system features:

- A trained machine learning model achieving **58.3% accuracy** on benchmark data
- An interactive web dashboard for real-time sentiment analysis
- Zero infrastructure requirements — runs entirely in the browser

## Key Metrics

| Metric | Value |
|--------|-------|
| Dataset Size | 59,899 tweets |
| Model Accuracy | 58.3% |
| Macro F1-Score | 55.8% |
| Classes | 3 (negative, neutral, positive) |
| Inference Location | Client-side (browser) |
| Server Requirements | None (static deployment) |

## Dataset

The project uses the **TweetEval Sentiment Benchmark** — a standardized, peer-reviewed evaluation dataset.

**Class Distribution (combined dataset):**

| Class | Count | Percentage |
|-------|-------|------------|
| Negative | 11,377 | 19.0% |
| Neutral | 27,479 | 45.9% |
| Positive | 21,043 | 35.1% |

**Test-Set Class Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| Negative | 3,972 | 32.3% |
| Neutral | 5,937 | 48.3% |
| Positive | 2,375 | 19.3% |

## Technical Highlights

### Data Integrity
- Used the **TweetEval benchmark** — a standardized, peer-reviewed evaluation dataset
- Strict train/validation/test split preservation with zero data leakage
- TF-IDF vectorizer trained exclusively on training data

### Model Selection
- **Logistic Regression** selected as primary model for interpretable, fast inference
- **Linear SVM** included as benchmark comparison
- Both models trained with `random_state=42` for reproducibility

### Deployment Architecture
- Static HTML/JS/CSS dashboard served from GitHub Pages
- In-browser JavaScript inference engine eliminates server costs
- Complete privacy — no data leaves the user's device
- Chart.js visualizations for executive-friendly data presentation

### Code Quality
- Shared prediction core (`predictor-core.js`) used by browser, Node.js CLI, and pytest
- Single canonical test set (`parity_test_cases.json`) consumed by all parity validators
- Python is the sole reference generator — JavaScript never writes reference files
- Pipeline automatically copies confusion matrices to deployment directory

## Dashboard Sections

### 1. Executive Overview
- Real-time KPI metrics (total samples, accuracy, F1-score, classes)
- Sentiment distribution donut chart (data-driven from Python-generated JSON)
- Dataset split visualization
- Top positive/negative sentiment-associated terms
- Negative sentiment alert (threshold >25%, reports actual test-set percentage)

### 2. Interactive Sentiment Analyzer
- Real-time text classification
- Confidence percentage display
- Probability breakdown across all classes
- No server calls — instant results
- Automated 25-case parity test runner
- Lexicon-based demo classifier

### 3. Model Performance & Business Insights
- Side-by-side model comparison table
- Class-wise performance visualization
- Confusion matrix images (auto-copied by pipeline)
- Data-derived business insights

## Testing & Validation

- **19 automated tests** all passing
- **25-sentence parity test** verifying Python/JavaScript prediction consistency
- **100% class-label agreement** between Python and JavaScript implementations
- **Probability differences = 0.000000** between implementations
- **Softmax verification:** `softmax(decision_function())` == `predict_proba()` with zero difference

## Business Applications

1. **Brand Monitoring** — Track real-time sentiment shifts around product launches
2. **Customer Feedback Analysis** — Automatically categorize support tickets and reviews
3. **Campaign Performance** — Measure sentiment impact of marketing campaigns
4. **Competitor Analysis** — Benchmark sentiment against competitor brands

## Limitations

- Accuracy constrained by inherent difficulty of short-text sentiment analysis
- Model trained on English-language tweets only
- Performance affected by class imbalance in training data
- Cannot capture sarcasm or context-dependent sentiment

## Deployment

The system is deployed as a static website on GitHub Pages, requiring zero infrastructure costs. The entire inference pipeline runs in the browser using exported model weights, ensuring complete data privacy.

**Local Verification:** `python -m http.server 8000 --directory docs`

## Technology Stack

- **ML Framework:** scikit-learn (Python)
- **Feature Engineering:** TF-IDF with bigrams
- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **Visualization:** Chart.js
- **Deployment:** GitHub Pages (static)
