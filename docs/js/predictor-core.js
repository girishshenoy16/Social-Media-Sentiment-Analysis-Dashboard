/**
 * predictor-core.js — Shared TF-IDF + LR prediction core.
 * Browser: exports via window.SentimentPredictorCore
 * Node.js: exports via module.exports
 * No I/O, no model loading — pure computation only.
 */
(function (exports) {
    "use strict";

    function cleanText(text) {
        if (typeof text !== "string") return "";
        text = text.toLowerCase();
        text = text.replace(/https?:\/\/\S+|www\.\S+/g, "");
        text = text.replace(/@\w+/g, "");
        text = text.replace(/[^a-zA-Z\s]/g, "");
        text = text.replace(/\s+/g, " ").trim();
        return text;
    }

    function generateNgrams(text) {
        var words = text.split(/\s+/).filter(Boolean);
        var unigrams = words;
        var bigrams = [];
        for (var i = 0; i < words.length - 1; i++) {
            bigrams.push(words[i] + " " + words[i + 1]);
        }
        return unigrams.concat(bigrams);
    }

    function vectorize(text, modelData) {
        var cleaned = cleanText(text);
        var ngrams = generateNgrams(cleaned);
        var vocabulary = modelData.vocabulary;
        var idf = modelData.idf;
        var maxFeatures = modelData.max_features;
        var vector = new Float64Array(maxFeatures);

        var tfCounts = {};
        for (var i = 0; i < ngrams.length; i++) {
            var ngram = ngrams[i];
            if (ngram in vocabulary) {
                var idx = vocabulary[ngram];
                if (idx < maxFeatures) {
                    tfCounts[idx] = (tfCounts[idx] || 0) + 1;
                }
            }
        }

        if (modelData.sublinear_tf) {
            for (var idx in tfCounts) {
                vector[idx] = 1 + Math.log(tfCounts[idx]);
            }
        } else {
            var total = ngrams.length || 1;
            for (var idx in tfCounts) {
                vector[idx] = tfCounts[idx] / total;
            }
        }

        for (var i = 0; i < maxFeatures; i++) {
            if (vector[i] > 0) vector[i] *= idf[i];
        }

        if (modelData.norm === "l2") {
            var normVal = 0;
            for (var i = 0; i < maxFeatures; i++) normVal += vector[i] * vector[i];
            normVal = Math.sqrt(normVal);
            if (normVal > 0) {
                for (var i = 0; i < maxFeatures; i++) vector[i] /= normVal;
            }
        }

        return vector;
    }

    function softmax(values) {
        var maxVal = Math.max.apply(null, values);
        var exps = values.map(function (v) { return Math.exp(v - maxVal); });
        var sumExps = exps.reduce(function (a, b) { return a + b; }, 0);
        return exps.map(function (e) { return e / sumExps; });
    }

    function predict(text, modelData) {
        var vector = vectorize(text, modelData);
        var coefficients = modelData.coefficients;
        var intercept = modelData.intercept;
        var classLabels = modelData.class_labels;
        var classes = modelData.classes;

        var decisions = [];
        for (var c = 0; c < classes.length; c++) {
            var decision = intercept[c];
            for (var i = 0; i < vector.length; i++) {
                decision += coefficients[c][i] * vector[i];
            }
            decisions.push(decision);
        }

        var probs = softmax(decisions);
        var maxProb = -1;
        var predIdx = 0;
        for (var i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                predIdx = i;
            }
        }

        var classId = classes[predIdx];
        var classLabel = classLabels[String(classId)];

        var probabilities = {};
        for (var i = 0; i < classes.length; i++) {
            probabilities[classLabels[String(classes[i])]] = probs[i];
        }

        return {
            predicted_class: predIdx,
            class_label: classLabel,
            probabilities: probabilities,
            confidence: maxProb
        };
    }

    exports.cleanText = cleanText;
    exports.generateNgrams = generateNgrams;
    exports.vectorize = vectorize;
    exports.softmax = softmax;
    exports.predict = predict;

})(typeof window !== "undefined" ? (window.SentimentPredictorCore = {}) : exports);
