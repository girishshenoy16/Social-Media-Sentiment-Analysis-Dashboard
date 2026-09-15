/**
 * Dashboard.js - Chart.js visualizations and metric rendering
 */
(function () {
    "use strict";

    let dashboardData = null;

    async function loadDashboardData() {
        try {
            const response = await fetch("data/dashboard_data.json");
            if (!response.ok) throw new Error("Failed to load dashboard data");
            dashboardData = await response.json();
            return dashboardData;
        } catch (e) {
            console.error("Dashboard data loading error:", e);
            return null;
        }
    }

    function renderKPIs(data) {
        if (!data) return;
        var stats = data.dataset_stats;
        var lrMetrics = data.metrics.logistic_regression;

        document.getElementById("kpi-total").textContent = stats.total_count.toLocaleString();
        document.getElementById("kpi-accuracy").textContent = (lrMetrics.accuracy * 100).toFixed(1) + "%";
        document.getElementById("kpi-f1").textContent = (lrMetrics.macro_f1 * 100).toFixed(1) + "%";
        document.getElementById("kpi-classes").textContent = stats.num_classes;
    }

    function renderSentimentDonut(data) {
        if (!data) return;
        var stats = data.dataset_stats;
        var dist = stats.class_distribution;

        var labels = ["Negative", "Neutral", "Positive"];
        var counts = [
            dist["negative"] || 0,
            dist["neutral"] || 0,
            dist["positive"] || 0
        ];

        var ctx = document.getElementById("chart-sentiment-donut").getContext("2d");
        new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: ["#E74C3C", "#F39C12", "#2ECC71"],
                    borderColor: "#FFFFFF",
                    borderWidth: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: "#5A6170",
                            padding: 16,
                            font: { size: 13 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                var total = context.dataset.data.reduce(function (a, b) { return a + b; }, 0);
                                var pct = ((context.parsed / total) * 100).toFixed(1);
                                return context.label + ": " + context.parsed.toLocaleString() + " (" + pct + "%)";
                            }
                        }
                    }
                }
            }
        });
    }

    function renderDatasetSplit(data) {
        if (!data) return;
        var stats = data.dataset_stats;

        var ctx = document.getElementById("chart-dataset-split").getContext("2d");
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Training", "Validation", "Test"],
                datasets: [{
                    label: "Samples",
                    data: [stats.train_count, stats.val_count, stats.test_count],
                    backgroundColor: ["#0078D4", "#2ECC71", "#F39C12"],
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: "#DDE1E8" },
                        ticks: { color: "#5A6170" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#5A6170" }
                    }
                }
            }
        });
    }

    function renderTopTerms(data) {
        if (!data || !data.metrics) return;

        var webModel = null;
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "data/web_model.json", false);
        xhr.send();
        if (xhr.status === 200) {
            webModel = JSON.parse(xhr.responseText);
        }

        if (!webModel || !webModel.top_terms) return;

        var posContainer = document.getElementById("positive-terms");
        var negContainer = document.getElementById("negative-terms");
        posContainer.innerHTML = "";
        negContainer.innerHTML = "";

        var positiveTerms = webModel.top_terms["positive"];
        var negativeTerms = webModel.top_terms["negative"];

        if (positiveTerms && positiveTerms.positive_terms) {
            positiveTerms.positive_terms.forEach(function (term) {
                var tag = document.createElement("span");
                tag.className = "term-tag positive";
                tag.textContent = term;
                posContainer.appendChild(tag);
            });
        }

        if (negativeTerms && negativeTerms.negative_terms) {
            negativeTerms.negative_terms.forEach(function (term) {
                var tag = document.createElement("span");
                tag.className = "term-tag negative";
                tag.textContent = term;
                negContainer.appendChild(tag);
            });
        }
    }

    function renderMetricsTable(data) {
        if (!data) return;
        var tbody = document.getElementById("metrics-tbody");
        tbody.innerHTML = "";

        var metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"];
        var labels = {
            accuracy: "Accuracy",
            macro_precision: "Macro Precision",
            macro_recall: "Macro Recall",
            macro_f1: "Macro F1-Score"
        };

        metrics.forEach(function (m) {
            var row = document.createElement("tr");
            var tdLabel = document.createElement("td");
            tdLabel.textContent = labels[m];
            var tdLR = document.createElement("td");
            tdLR.textContent = (data.metrics.logistic_regression[m] * 100).toFixed(2) + "%";
            var tdSVM = document.createElement("td");
            tdSVM.textContent = (data.metrics.linear_svm[m] * 100).toFixed(2) + "%";
            row.appendChild(tdLabel);
            row.appendChild(tdLR);
            row.appendChild(tdSVM);
            tbody.appendChild(row);
        });
    }

    function renderClassPerformanceChart(data) {
        if (!data) return;
        var lrMetrics = data.metrics.logistic_regression;
        var svmMetrics = data.metrics.linear_svm;

        var ctx = document.getElementById("chart-class-performance").getContext("2d");
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
                datasets: [
                    {
                        label: "Logistic Regression",
                        data: [
                            lrMetrics.accuracy,
                            lrMetrics.macro_precision,
                            lrMetrics.macro_recall,
                            lrMetrics.macro_f1
                        ],
                        backgroundColor: "#0078D4",
                        borderRadius: 4,
                    },
                    {
                        label: "Linear SVM",
                        data: [
                            svmMetrics.accuracy,
                            svmMetrics.macro_precision,
                            svmMetrics.macro_recall,
                            svmMetrics.macro_f1
                        ],
                        backgroundColor: "#2ECC71",
                        borderRadius: 4,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: "#5A6170" }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: { color: "#DDE1E8" },
                        ticks: {
                            color: "#5A6170",
                            callback: function (v) { return (v * 100) + "%"; }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#5A6170" }
                    }
                }
            }
        });
    }

    function renderConfusionMatrices() {
        var imgLR = document.getElementById("img-confusion-lr");
        var imgSVM = document.getElementById("img-confusion-svm");

        imgLR.src = "outputs/confusion_matrix_logistic_regression.png";
        imgSVM.src = "outputs/confusion_matrix_linear_svm.png";

        imgLR.onerror = function () {
            imgLR.alt = "Confusion matrix image not found. Run the pipeline to generate it.";
            imgLR.style.display = "none";
        };
        imgSVM.onerror = function () {
            imgSVM.alt = "Confusion matrix image not found. Run the pipeline to generate it.";
            imgSVM.style.display = "none";
        };
    }

    function renderBusinessInsights(data) {
        if (!data) return;
        var container = document.getElementById("business-insights");
        container.innerHTML = "";

        var lrMetrics = data.metrics.logistic_regression;
        var stats = data.dataset_stats;

        var insights = [
            {
                title: "Dataset Balance",
                text: "The dataset contains " + stats.total_count.toLocaleString() + " tweets across " + stats.num_classes + " sentiment classes. The neutral class is the largest class, accounting for approximately 45.9% of the dataset."
            },
            {
                title: "Model Performance",
                text: "Logistic Regression achieves " + (lrMetrics.accuracy * 100).toFixed(1) + "% accuracy with a macro F1-score of " + (lrMetrics.macro_f1 * 100).toFixed(1) + "%, demonstrating reasonable sentiment classification on short-form text."
            },
            {
                title: "Class Imbalance Impact",
                text: "The negative class has fewer training examples than neutral and also shows substantially lower recall, indicating a potential class-distribution effect."
            },
            {
                title: "Business Application",
                text: "The browser-deployed model can support sentiment analysis workflows such as customer feedback analysis, brand monitoring, and campaign evaluation when integrated with an appropriate data source."
            },
            {
                title: "Feature Engineering",
                text: "TF-IDF with unigrams and bigrams captures individual words and word pairs, allowing the model to represent phrases such as 'not good'."
            },
            {
                title: "Deployment Ready",
                text: "The exported JavaScript inference engine runs entirely in-browser with zero server requirements, enabling privacy-preserving sentiment analysis."
            }
        ];

        insights.forEach(function (insight) {
            var card = document.createElement("div");
            card.className = "insight-card";
            card.innerHTML = "<h4>" + insight.title + "</h4><p>" + insight.text + "</p>";
            container.appendChild(card);
        });
    }

    async function initDashboard() {
        var data = await loadDashboardData();
        if (!data) {
            console.error("Failed to load dashboard data");
            return;
        }

        renderKPIs(data);
        renderSentimentDonut(data);
        renderDatasetSplit(data);
        renderTopTerms(data);
        renderMetricsTable(data);
        renderClassPerformanceChart(data);
        renderConfusionMatrices();
        renderBusinessInsights(data);
    }

    document.addEventListener("DOMContentLoaded", function () {
        initDashboard();
    });

    window.DashboardModule = {
        initDashboard: initDashboard,
        loadDashboardData: loadDashboardData,
    };
})();
