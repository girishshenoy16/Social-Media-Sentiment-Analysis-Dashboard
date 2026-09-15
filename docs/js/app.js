/**
 * App.js - Navigation controller and interactive prediction event listeners
 */
(function () {
    "use strict";

    function initNavigation() {
        var navLinks = document.querySelectorAll(".nav-link");
        var sections = document.querySelectorAll(".dashboard-section");

        navLinks.forEach(function (link) {
            link.addEventListener("click", function (e) {
                e.preventDefault();

                var targetId = this.getAttribute("data-section");

                navLinks.forEach(function (l) { l.classList.remove("active"); });
                sections.forEach(function (s) { s.classList.remove("active"); });

                this.classList.add("active");
                var targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add("active");
                }
            });
        });
    }

    function initSentimentAnalyzer() {
        var analyzeBtn = document.getElementById("analyze-btn");
        var input = document.getElementById("sentiment-input");
        var resultDiv = document.getElementById("prediction-result");
        var errorDiv = document.getElementById("prediction-error");

        if (!analyzeBtn) return;

        analyzeBtn.addEventListener("click", function () {
            var text = input.value.trim();

            resultDiv.classList.add("hidden");
            errorDiv.classList.add("hidden");

            if (!text) {
                showError("Please enter some text to analyze.");
                return;
            }

            if (typeof SentimentPredictor === "undefined") {
                showError("Prediction engine not loaded. Please refresh the page.");
                return;
            }

            SentimentPredictor.predict(text)
                .then(function (result) {
                    if (result.predicted_class === -1) {
                        showError("Could not analyze empty input.");
                        return;
                    }
                    showResult(result);
                    trackPrediction(result);
                })
                .catch(function (err) {
                    showError("Prediction error: " + (err.message || err));
                });
        });

        input.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                analyzeBtn.click();
            }
        });
    }

    function showResult(result) {
        var resultDiv = document.getElementById("prediction-result");
        var errorDiv = document.getElementById("prediction-error");

        errorDiv.classList.add("hidden");
        resultDiv.classList.remove("hidden");

        var labelEl = document.getElementById("pred-label");
        var confidenceEl = document.getElementById("pred-confidence");

        labelEl.textContent = result.class_label.charAt(0).toUpperCase() + result.class_label.slice(1);
        labelEl.style.color = getSentimentColor(result.class_label);
        confidenceEl.textContent = (result.confidence * 100).toFixed(1) + "%";

        var probBars = document.getElementById("prob-bars");
        probBars.innerHTML = "";

        var labels = Object.keys(result.probabilities);
        labels.sort();

        labels.forEach(function (label) {
            var prob = result.probabilities[label];
            var row = document.createElement("div");
            row.className = "prob-bar-row";

            var labelSpan = document.createElement("span");
            labelSpan.className = "prob-bar-label";
            labelSpan.textContent = label;

            var track = document.createElement("div");
            track.className = "prob-bar-track";

            var fill = document.createElement("div");
            fill.className = "prob-bar-fill";
            fill.style.width = (prob * 100) + "%";
            fill.style.backgroundColor = getSentimentColor(label);

            track.appendChild(fill);

            var valueSpan = document.createElement("span");
            valueSpan.className = "prob-bar-value";
            valueSpan.textContent = (prob * 100).toFixed(1) + "%";

            row.appendChild(labelSpan);
            row.appendChild(track);
            row.appendChild(valueSpan);
            probBars.appendChild(row);
        });
    }

    function showError(message) {
        var errorDiv = document.getElementById("prediction-error");
        var msgSpan = document.getElementById("error-message");
        errorDiv.classList.remove("hidden");
        msgSpan.textContent = message;
    }

    function getSentimentColor(label) {
        switch (label.toLowerCase()) {
            case "positive": return "#2ECC71";
            case "neutral": return "#F39C12";
            case "negative": return "#E74C3C";
            default: return "#B0B0B0";
        }
    }

    function initPredictor() {
        if (typeof SentimentPredictor !== "undefined" && SentimentPredictor.loadModel) {
            SentimentPredictor.loadModel()
                .then(function () {
                    console.log("Sentiment predictor model loaded successfully");
                })
                .catch(function (err) {
                    console.error("Failed to load predictor model:", err);
                });
        }
    }

    function initLexiconDemo() {
        var lexiconBtn = document.getElementById("lexicon-btn");
        var lexiconInput = document.getElementById("lexicon-input");
        var lexiconResult = document.getElementById("lexicon-result");
        var lexiconLabel = document.getElementById("lexicon-label");
        var lexiconScore = document.getElementById("lexicon-score");

        if (!lexiconBtn) return;

        function runLexicon() {
            var text = lexiconInput.value.trim();
            if (!text || typeof SentimentPredictor === "undefined" || !SentimentPredictor.lexiconPredict) return;

            var result = SentimentPredictor.lexiconPredict(text);
            lexiconLabel.textContent = result.label.charAt(0).toUpperCase() + result.label.slice(1);
            lexiconLabel.style.color = getSentimentColor(result.label);
            lexiconScore.textContent = "Score: " + result.score.toFixed(2) + " | Confidence: " + (result.confidence * 100).toFixed(0) + "%";
            lexiconResult.style.display = "block";
        }

        lexiconBtn.addEventListener("click", runLexicon);
        lexiconInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") runLexicon();
        });
    }

    var ALERT_WINDOW_SIZE = 50;
    var predictionHistory = [];

    function trackPrediction(result) {
        if (result && result.class_label) {
            predictionHistory.push(result.class_label);
            if (predictionHistory.length > ALERT_WINDOW_SIZE) {
                predictionHistory.shift();
            }
            updateNegativeAlert();
        }
    }

    function updateNegativeAlert() {
        if (predictionHistory.length === 0) return;

        var negCount = 0;
        for (var i = 0; i < predictionHistory.length; i++) {
            if (predictionHistory[i] === "negative") negCount++;
        }
        var negPct = (negCount / predictionHistory.length) * 100;

        if (negPct > 25) {
            var banner = document.getElementById("alert-banner");
            var msg = document.getElementById("alert-message");
            if (banner && msg) {
                msg.textContent = "Negative sentiment exceeds the 25% threshold — " + negPct.toFixed(1) + "% of the last " + predictionHistory.length + " tweets are classified as negative. Immediate attention recommended.";
                banner.classList.add("visible");
            }
        }
    }

    function checkNegativeAlert() {
        var banner = document.getElementById("alert-banner");
        if (banner) banner.classList.remove("visible");
    }

    var TEST_SENTENCES = [
        "I love this product! It's amazing!",
        "Best day ever! So happy!",
        "Great work, really impressive!",
        "Absolutely wonderful experience today",
        "The movie was fantastic and thrilling",
        "The weather is cloudy today.",
        "Meeting starts at 3pm.",
        "Just finished reading a book.",
        "The report was released yesterday.",
        "Coffee is served every morning.",
        "This is terrible and disappointing.",
        "Worst experience of my life.",
        "I hate waiting in long lines.",
        "The service was awful and rude.",
        "Very bad quality product.",
        "Check https://example.com for details",
        "@user Thanks for sharing!",
        "Visit www.test.com today",
        "Wow!!! This is incredible...",
        "Hello... are you there?",
        "So, what do you think?",
        "THIS IS SO GOOD!",
        "STOP doing that!!!",
        "Not bad, not great either",
        "Pretty decent overall nothing special",
    ];

    var pythonReferences = null;
    var referenceLoadError = false;

    function loadReferences(callback) {
        if (pythonReferences !== null) { callback(); return; }
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "data/parity_reference.json", true);
        xhr.onload = function () {
            if (xhr.status === 200) {
                try {
                    pythonReferences = JSON.parse(xhr.responseText);
                } catch (e) {
                    referenceLoadError = true;
                }
            } else {
                referenceLoadError = true;
            }
            callback();
        };
        xhr.onerror = function () {
            referenceLoadError = true;
            callback();
        };
        xhr.send();
    }

    function initTestCases() {
        var runBtn = document.getElementById("run-tests-btn");
        if (!runBtn) return;

        runBtn.addEventListener("click", function () {
            if (typeof SentimentPredictor === "undefined" || !SentimentPredictor.predict) {
                alert("Prediction engine not loaded. Please refresh the page.");
                return;
            }

            runBtn.disabled = true;
            runBtn.textContent = "Running...";

            loadReferences(function () {
                var summaryDiv = document.getElementById("test-summary");
                var detailsDiv = document.getElementById("test-details");
                var tbody = document.getElementById("test-results-body");
                var toggleBtn = document.getElementById("toggle-details-btn");

                summaryDiv.classList.remove("hidden");
                detailsDiv.classList.add("hidden");
                tbody.innerHTML = "";
                toggleBtn.textContent = "Show Details";

                if (referenceLoadError || !pythonReferences) {
                    document.getElementById("tc-total").textContent = TEST_SENTENCES.length;
                    document.getElementById("tc-passed").textContent = "-";
                    document.getElementById("tc-failed").textContent = "-";
                    document.getElementById("tc-agreement").textContent = "ERR";
                    document.getElementById("tc-maxdiff").textContent = "ERR";
                    var overallEl = document.getElementById("tc-overall");
                    overallEl.textContent = "ERR";
                    overallEl.className = "test-kpi-val tc-overall-fail";
                    runBtn.disabled = false;
                    runBtn.textContent = "Run Test Cases";
                    alert("Could not load Python reference from data/parity_reference.json. Run 'python main.py' to generate it.");
                    return;
                }

                var passed = 0, failed = 0, maxProbDiff = 0;
                var results = [];

                var promises = TEST_SENTENCES.map(function (sentence, i) {
                    return SentimentPredictor.predict(sentence).then(function (jsResult) {
                        var ref = pythonReferences ? pythonReferences[i] : null;
                        var pyLabel = ref ? ref.class_label : null;
                        var agreement = pyLabel ? (jsResult.class_label === pyLabel) : null;
                        var probDiff = 0;

                        if (ref && ref.probabilities) {
                            for (var label in ref.probabilities) {
                                var diff = Math.abs((ref.probabilities[label] || 0) - (jsResult.probabilities[label] || 0));
                                if (diff > probDiff) probDiff = diff;
                            }
                        }
                        if (probDiff > maxProbDiff) maxProbDiff = probDiff;
                        if (agreement === true) passed++;
                        else if (agreement === false) failed++;

                        results.push({
                            index: i + 1,
                            sentence: sentence,
                            jsLabel: jsResult.class_label,
                            pyLabel: pyLabel,
                            agreement: agreement,
                            confidence: jsResult.confidence,
                            probDiff: probDiff,
                        });
                    });
                });

                Promise.all(promises).then(function () {
                    var total = TEST_SENTENCES.length;
                    var agreementPct = pythonReferences ? ((passed / total) * 100).toFixed(1) + "%" : "N/A";
                    var overallPass = pythonReferences ? (failed === 0 && maxProbDiff < 1e-3) : false;

                    document.getElementById("tc-total").textContent = total;
                    document.getElementById("tc-passed").textContent = passed;
                    document.getElementById("tc-failed").textContent = failed;
                    document.getElementById("tc-agreement").textContent = agreementPct;
                    document.getElementById("tc-maxdiff").textContent = maxProbDiff.toFixed(6);
                    var overallEl = document.getElementById("tc-overall");
                    overallEl.textContent = overallPass ? "PASS" : "FAIL";
                    overallEl.className = "test-kpi-val " + (overallPass ? "tc-overall-pass" : "tc-overall-fail");

                    results.sort(function (a, b) { return a.index - b.index; });
                    results.forEach(function (r) {
                        var row = document.createElement("tr");
                        var agreeText = r.agreement === true ? "YES" : r.agreement === false ? "NO" : "N/A";
                        var agreeClass = r.agreement === true ? "tc-pass" : r.agreement === false ? "tc-fail" : "";
                        row.innerHTML =
                            "<td>" + r.index + "</td>" +
                            "<td>" + escapeHtml(r.sentence.substring(0, 55)) + (r.sentence.length > 55 ? "..." : "") + "</td>" +
                            "<td>" + capitalize(r.jsLabel) + "</td>" +
                            "<td class=\"" + agreeClass + "\">" + agreeText + "</td>" +
                            "<td>" + (r.confidence * 100).toFixed(1) + "%</td>" +
                            "<td>" + r.probDiff.toFixed(6) + "</td>";
                        tbody.appendChild(row);
                    });

                    runBtn.disabled = false;
                    runBtn.textContent = "Run Test Cases";
                });
            });
        });

        var toggleBtn = document.getElementById("toggle-details-btn");
        if (toggleBtn) {
            toggleBtn.addEventListener("click", function () {
                var detailsDiv = document.getElementById("test-details");
                var isHidden = detailsDiv.classList.contains("hidden");
                detailsDiv.classList.toggle("hidden");
                this.textContent = isHidden ? "Hide Details" : "Show Details";
            });
        }
    }

    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    document.addEventListener("DOMContentLoaded", function () {
        initNavigation();
        initSentimentAnalyzer();
        initPredictor();
        initLexiconDemo();
        initTestCases();
        checkNegativeAlert();
    });
})();
