/**
 * In-browser TF-IDF + Logistic Regression prediction engine.
 * Loads web_model.json once, then performs local predictions.
 * Uses shared predictor-core.js for all prediction math.
 */
(function (global) {
    "use strict";

    var core = global.SentimentPredictorCore;
    var modelData = null;

    /**
     * Load model artifacts from web_model.json
     */
    function loadModel() {
        if (modelData) return Promise.resolve(modelData);
        return fetch("data/web_model.json")
            .then(function (response) {
                if (!response.ok) throw new Error("Failed to load model: " + response.status);
                return response.json();
            })
            .then(function (data) {
                modelData = data;
                return modelData;
            });
    }

    /**
     * Predict sentiment for a given text.
     */
    function predict(text) {
        if (!modelData) {
            return Promise.reject(new Error("Model not loaded. Call loadModel() first."));
        }

        if (!text || text.trim().length === 0) {
            return Promise.resolve({
                text: text,
                predicted_class: -1,
                class_label: "unknown",
                probabilities: {},
                confidence: 0,
            });
        }

        var result = core.predict(text, modelData);
        result.text = text;
        return Promise.resolve(result);
    }

    /**
     * Predict multiple texts.
     */
    function predictBatch(texts) {
        return Promise.all(texts.map(function (t) { return predict(t); }));
    }

    // --- VADER-style Lexicon for demo classification ---
    var LEXICON = {
        "positive": {
            "good": 1.9, "great": 2.6, "excellent": 3.0, "amazing": 3.0, "awesome": 3.0,
            "fantastic": 2.9, "wonderful": 2.8, "love": 2.5, "loved": 2.5, "loving": 2.5,
            "happy": 2.8, "happiness": 2.8, "joy": 2.7, "joyful": 2.7, "best": 2.8,
            "better": 1.5, "nice": 1.8, "beautiful": 2.8, "brilliant": 2.9, "perfect": 3.0,
            "fun": 2.2, "enjoy": 2.1, "enjoyed": 2.1, "recommend": 1.8, "recommended": 1.8,
            "thanks": 1.2, "thank": 1.2, "thankful": 2.0, "grateful": 2.2, "appreciate": 1.8,
            "excited": 2.5, "exciting": 2.5, "impressive": 2.4, "impressed": 2.4,
            "superb": 2.9, "outstanding": 2.8, "remarkable": 2.5, "phenomenal": 3.0,
            "terrific": 2.7, "fabulous": 2.7, "magnificent": 2.8, "splendid": 2.7,
            "glad": 2.0, "pleased": 2.0, "delighted": 2.6, "thrilled": 2.8,
            "satisfied": 1.8, "content": 1.2, "peaceful": 1.8, "calm": 1.2,
            "hope": 1.5, "hopeful": 1.8, "inspire": 2.0, "inspired": 2.0, "inspiring": 2.2,
            "success": 2.2, "successful": 2.3, "win": 2.0, "won": 2.0, "winning": 2.2,
            "favor": 1.2, "positive": 1.5, "upbeat": 1.8, "cheerful": 2.0,
            "laugh": 1.8, "laughing": 2.0, "smile": 1.8, "smiling": 1.8,
            "kind": 1.5, "generous": 1.8, "caring": 1.8, "friendly": 1.8, "welcoming": 1.5,
            "support": 1.2, "supported": 1.2, "supportive": 1.5,
            "quality": 1.2, "reliable": 1.5, "trust": 1.5, "trustworthy": 1.8,
            "innovative": 1.8, "creative": 1.5, "smart": 1.2, "wise": 1.2,
            "helpful": 1.8, "help": 1.0, "easy": 1.2, "simple": 1.0,
            "free": 0.8, "new": 0.5, "fresh": 1.0, "clean": 1.0,
            "strong": 1.2, "powerful": 1.5, "fast": 1.0, "quick": 1.0,
            "cool": 1.2, "hot": 0.5, "wow": 2.0, "yay": 2.5,
            "bravo": 2.5, "congrats": 2.2, "congratulations": 2.5, "celebrate": 2.0,
            "progress": 1.5, "improve": 1.2, "improved": 1.5, "improvement": 1.5,
            "opportunity": 1.2, "benefit": 1.2, "advantage": 1.2,
            "passion": 1.5, "passionate": 1.8, "enthusiasm": 1.8, "eager": 1.5,
            "confident": 1.5, "confidence": 1.5, "proud": 1.8, "pride": 1.5,
            "correct": 1.0, "right": 0.8, "agreed": 1.0, "yes": 0.8,
            "absolutely": 1.8, "definitely": 1.5, "certainly": 1.2, "indeed": 1.0,
            "gorgeous": 2.8, "stunning": 2.7, "charming": 2.0, "elegant": 2.0,
            "hilarious": 2.5, "funny": 1.8, "clever": 1.5, "genius": 2.5,
            "brave": 1.5, "courage": 1.5, "hero": 2.0, "heroic": 2.2,
            "miracle": 2.5, "blessed": 2.2, "grace": 1.5, "elegant": 2.0,
            "rare": 0.8, "special": 1.2, "unique": 1.2, "original": 1.0,
            "classic": 1.0, "legendary": 2.2, "iconic": 1.8,
            "safe": 1.0, "secure": 1.2, "healthy": 1.5, "fit": 0.8,
            "smooth": 1.0, "seamless": 1.5, "flawless": 2.8, "spotless": 2.0,
            "ready": 0.5, "set": 0.5, "go": 0.5, "done": 0.8,
            "worth": 1.2, "valuable": 1.5, "precious": 1.5, "priceless": 2.0,
            "welcome": 1.2, "invited": 1.0, "included": 1.0, "accepted": 1.2,
            "forgive": 1.0, "sorry": 0.5, "apologize": 0.5, "regret": -0.5,
            "relief": 1.5, "relieved": 1.5, "relax": 1.2, "relaxed": 1.2,
            "trust": 1.5, "believe": 1.2, "faith": 1.5, "loyal": 1.5,
            "united": 1.2, "together": 1.0, "team": 0.8, "community": 1.2
        },
        "negative": {
            "bad": -1.9, "terrible": -3.0, "horrible": -3.0, "awful": -3.0, "dreadful": -2.9,
            "worst": -3.0, "worse": -2.0, "hate": -2.8, "hated": -2.8, "hating": -2.8,
            "angry": -2.5, "anger": -2.5, "furious": -3.0, "rage": -2.8, "mad": -2.0,
            "sad": -2.0, "sadness": -2.0, "unhappy": -2.0, "depressed": -2.8, "depression": -2.8,
            "disappointed": -2.2, "disappointing": -2.2, "disappointment": -2.2,
            "fail": -2.0, "failed": -2.0, "failure": -2.5, "failing": -1.5,
            "wrong": -1.5, "error": -1.2, "mistake": -1.5, "bug": -1.2, "broken": -2.0,
            "stupid": -2.5, "dumb": -2.0, "idiot": -2.8, "fool": -2.0, "foolish": -2.0,
            "ugly": -2.5, "disgusting": -2.8, "gross": -2.0, "nasty": -2.5,
            "boring": -1.8, "bored": -1.5, "dull": -1.5, "tedious": -1.8, "tiresome": -1.5,
            "annoying": -2.0, "annoyed": -2.0, "irritating": -2.2, "frustrating": -2.2,
            "pain": -2.0, "painful": -2.5, "hurt": -2.2, "suffering": -2.5, "agony": -2.8,
            "fear": -2.0, "afraid": -2.0, "scared": -2.2, "terrified": -2.8, "panic": -2.5,
            "anxious": -1.8, "anxiety": -2.0, "worry": -1.8, "worried": -1.8, "nervous": -1.5,
            "threat": -2.0, "danger": -2.0, "dangerous": -2.2, "risk": -1.2, "risky": -1.5,
            "toxic": -2.5, "poison": -2.5, "contaminate": -2.2, "infect": -2.0,
            "war": -2.5, "fight": -1.5, "conflict": -1.8, "attack": -2.0, "destroy": -2.5,
            "kill": -2.8, "death": -2.8, "dead": -2.5, "die": -2.5, "dying": -2.5,
            "cry": -1.8, "crying": -2.0, "tears": -1.5, "weep": -2.0,
            "lonely": -2.0, "alone": -1.2, "isolated": -1.8, "abandoned": -2.5,
            "lost": -1.5, "confused": -1.2, "uncertain": -1.0, "doubt": -1.2,
            "lie": -2.0, "lied": -2.2, "lying": -2.2, "cheat": -2.2, "cheated": -2.5,
            "steal": -2.2, "stolen": -2.5, "rob": -2.2, "robbed": -2.5,
            "racist": -2.8, "sexist": -2.8, "biased": -1.8, "discrimination": -2.5,
            "abuse": -2.8, "abused": -2.8, "harass": -2.5, "harassed": -2.5, "bully": -2.5,
            "ignore": -1.2, "ignored": -1.8, "neglect": -2.0, "neglected": -2.2,
            "reject": -1.8, "rejected": -2.2, "denied": -1.5, "refuse": -1.5,
            "problem": -1.2, "issue": -1.0, "trouble": -1.5, "crisis": -2.2,
            "corrupt": -2.5, "fraud": -2.5, "scam": -2.5, "fake": -2.0, "phony": -2.0,
            "expensive": -1.0, "overpriced": -1.5, "waste": -1.5, "worthless": -2.5,
            "slow": -1.0, "late": -1.0, "delay": -1.2, "waiting": -0.8, "stuck": -1.5,
            "hard": -0.8, "difficult": -1.0, "impossible": -2.0, "complex": -0.5,
            "loud": -0.8, "noise": -1.0, "harsh": -1.5,
            "cold": -0.8, "dark": -0.8, "dim": -0.5, "gloomy": -1.5,
            "mess": -1.2, "dirty": -1.5, "filthy": -2.0, "stain": -1.0,
            "stupid": -2.5, "ridiculous": -2.0, "absurd": -1.8, "nonsense": -1.8,
            "pathetic": -2.5, "miserable": -2.5, "hopeless": -2.5, "desperate": -2.2,
            "regret": -2.0, "regretful": -2.2, "ashamed": -2.2, "embarrassed": -1.8,
            "useless": -2.2, "pointless": -2.0, "meaningless": -1.8,
            "sick": -1.5, "ill": -1.5, "disease": -2.0, "pain": -2.0,
            "tired": -1.0, "exhausted": -1.8, "drained": -1.5, "weary": -1.5,
            "jealous": -1.8, "envy": -1.5, "greedy": -1.8, "selfish": -2.0,
            "rude": -2.0, "impolite": -1.8, "mean": -2.0, "cruel": -2.5,
            "crazy": -1.5, "insane": -1.8, "wild": -0.5, "unstable": -1.5,
            "weak": -1.2, "fragile": -1.0, "vulnerable": -1.2, "helpless": -2.0,
            "debt": -1.5, "loan": -0.8, "tax": -0.8, "cost": -0.5, "loss": -1.5,
            "lose": -1.2, "losing": -1.5, "lost": -1.5, "defeat": -1.8
        }
    };

    function lexiconPredict(text) {
        var cleaned = core.cleanText(text);
        var words = cleaned.split(/\s+/).filter(Boolean);
        if (words.length === 0) {
            return { label: "neutral", score: 0, confidence: 0 };
        }

        var posScore = 0, negScore = 0, matched = 0;
        for (var i = 0; i < words.length; i++) {
            var w = words[i];
            if (LEXICON.positive[w]) { posScore += LEXICON.positive[w]; matched++; }
            if (LEXICON.negative[w]) { negScore += Math.abs(LEXICON.negative[w]); matched++; }
        }

        var total = posScore + negScore;
        var label, confidence;
        if (total === 0) {
            label = "neutral";
            confidence = 0.5;
        } else {
            var net = (posScore - negScore) / total;
            if (net > 0.05) { label = "positive"; confidence = 0.5 + net * 0.5; }
            else if (net < -0.05) { label = "negative"; confidence = 0.5 + Math.abs(net) * 0.5; }
            else { label = "neutral"; confidence = 0.5 + (1 - Math.abs(net)) * 0.3; }
        }

        return { label: label, score: posScore - negScore, confidence: Math.min(confidence, 0.99) };
    }

    // Public API
    global.SentimentPredictor = {
        loadModel: loadModel,
        predict: predict,
        predictBatch: predictBatch,
        cleanText: core.cleanText,
        lexiconPredict: lexiconPredict,
    };

})(typeof window !== "undefined" ? window : global);
