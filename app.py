from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# ---------------------------
# DATASET
# ---------------------------
data = {
    "text": [
        "satellite monitors climate change",
        "global warming tracked from space",
        "weather prediction using satellite",
        "space data helps disaster management",

        "satellite improves crop yield",
        "remote sensing for farming",
        "space tech helps agriculture",

        "deforestation detected via satellite",
        "forest monitoring from space",
        "wildlife tracking using satellite",

        "new space mission launched",
        "rocket launch successful",
        "satellite development for research",
        "space exploration mission",

        "space debris increasing risk",
        "satellite collision danger",
        "launch failure caused damage",
        "rocket explosion failure"
    ],

    "sdg": [
        "SDG 13","SDG 13","SDG 13","SDG 13",
        "SDG 2","SDG 2","SDG 2",
        "SDG 15","SDG 15","SDG 15",
        "SDG 9","SDG 9","SDG 9","SDG 9",
        "Negative","Negative","Negative","Negative"
    ]
}

df = pd.DataFrame(data)

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# ---------------------------
# MODEL TRAINING
# ---------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(df["text"], df["sdg"])

# ---------------------------
# FUNCTIONS
# ---------------------------
def predict_with_confidence(text):
    text = clean_text(text)
    probs = model.predict_proba([text])[0]
    classes = model.classes_

    result = dict(zip(classes, probs))
    prediction = model.predict([text])[0]

    return prediction, result


def extract_keywords(text):
    text = clean_text(text)

    vectorizer = model.named_steps['tfidf']
    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.toarray()[0]

    keywords = sorted(
        [(feature_names[i], scores[i]) for i in range(len(scores))],
        key=lambda x: x[1],
        reverse=True
    )

    return [word for word, score in keywords[:3]]


def impact_type(text):
    text = text.lower()

    positive_words = ["help", "improve", "monitor", "support"]
    negative_words = ["risk", "damage", "failure", "debris", "explosion"]

    for word in negative_words:
        if word in text:
            return "Negative ❌"

    for word in positive_words:
        if word in text:
            return "Positive ✅"

    return "Neutral ⚪"


def generate_insight(pred):
    insights = {
        "SDG 13": "Supports climate monitoring and disaster prevention",
        "SDG 2": "Improves agriculture and food security",
        "SDG 15": "Helps protect forests and biodiversity",
        "SDG 9": "Promotes innovation and space technology development",
        "Negative": "Indicates environmental or technological risk"
    }

    return insights.get(pred, "No insight available")


def rule_based_boost(text):
    text = text.lower()

    if "mission" in text or "launch" in text:
        return "SDG 9"
    if "climate" in text or "weather" in text:
        return "SDG 13"
    if "crop" in text or "agriculture" in text:
        return "SDG 2"

    return None


# ---------------------------
# API ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    rule = rule_based_boost(text)

    if rule:
        pred = rule
        confidence = 1.0
        probs = {}
    else:
        pred, probs = predict_with_confidence(text)
        confidence = max(probs.values())

    keywords = extract_keywords(text)
    impact = impact_type(text)
    insight = generate_insight(pred)

    return jsonify({
        "sdg": pred,
        "confidence": round(confidence, 2),
        "keywords": keywords,
        "impact": impact,
        "insight": insight
    })


# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)