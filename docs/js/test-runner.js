#!/usr/bin/env node
/**
 * test-runner.js — Node.js CLI for 25-case Python ↔ JavaScript parity validation.
 * Run: node docs/js/test-runner.js
 *
 * Loads web_model.json, runs 25 canonical sentences through the shared TF-IDF + LR
 * predictor, compares against Python reference predictions, and prints a compact report.
 */

var fs = require("fs");
var path = require("path");
var core = require("./predictor-core");

var modelPath = path.join(__dirname, "..", "data", "web_model.json");
var modelData = JSON.parse(fs.readFileSync(modelPath, "utf8"));

var testCasesPath = path.join(__dirname, "..", "..", "tests", "parity_test_cases.json");
var TEST_SENTENCES = JSON.parse(fs.readFileSync(testCasesPath, "utf8"));

var PYTHON_REFERENCES = null;
var refPath = path.join(__dirname, "..", "..", "tests", "python_reference.json");
if (fs.existsSync(refPath)) {
    PYTHON_REFERENCES = JSON.parse(fs.readFileSync(refPath, "utf8"));
}

function runTests() {
    var results = [];
    var passed = 0;
    var failed = 0;
    var maxProbDiff = 0;

    for (var i = 0; i < TEST_SENTENCES.length; i++) {
        var sentence = TEST_SENTENCES[i];
        var jsResult = core.predict(sentence, modelData);

        var ref = PYTHON_REFERENCES ? PYTHON_REFERENCES[i] : null;
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
            sentence: sentence.substring(0, 50) + (sentence.length > 50 ? "..." : ""),
            js_label: jsResult.class_label,
            py_label: pyLabel,
            agreement: agreement,
            confidence: jsResult.confidence,
            prob_diff: probDiff,
        });
    }

    var total = TEST_SENTENCES.length;
    var agreementPct = PYTHON_REFERENCES ? ((passed / total) * 100).toFixed(1) + "%" : "N/A (no reference)";
    var overallPass = PYTHON_REFERENCES ? (failed === 0 && maxProbDiff < 1e-3) : "NO_REFERENCE";

    console.log("\n=== Test Case Results ===");
    console.log("Total:   " + total);
    console.log("Passed:  " + passed);
    console.log("Failed:  " + failed);
    console.log("Agreement: " + agreementPct);
    console.log("Max Prob Diff: " + maxProbDiff.toFixed(6));
    console.log("Overall: " + (overallPass === true ? "PASS" : overallPass === false ? "FAIL" : "NO_REFERENCE"));

    if (failed > 0) {
        console.log("\nFailed cases:");
        results.forEach(function (r) {
            if (r.agreement === false) {
                console.log("  #" + r.index + " JS=" + r.js_label + " PY=" + r.py_label + " \"" + r.sentence + "\"");
            }
        });
    }

    return { total: total, passed: passed, failed: failed, agreement: agreementPct, maxProbDiff: maxProbDiff, overall: overallPass, results: results };
}

var report = runTests();
process.exit(report.failed > 0 ? 1 : 0);
