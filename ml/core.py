import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import IsolationForest

NEEDS = {"Groceries", "Bills", "Transport", "Health", "Education"}


def clean_text(text):
    text = re.sub(r"[^a-z\s']", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def validate_transaction(amount, description):
    if amount is None or float(amount) <= 0:
        raise ValueError("Amount must be greater than zero")
    if not clean_text(description):
        raise ValueError("Description cannot be empty")
    return True


class Categoriser:
    """TF-IDF (character n-grams) + Logistic Regression."""
    def __init__(self, threshold=0.25):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.clf = LogisticRegression(max_iter=1000)
        self.threshold = threshold

    def fit(self, texts, labels):
        X = self.vec.fit_transform([clean_text(t) for t in texts])
        self.clf.fit(X, labels)
        return self

    def predict(self, text):
        text = clean_text(text)
        if not text:
            return "Uncategorised", 0.0
        p = self.clf.predict_proba(self.vec.transform([text]))[0]
        i = int(p.argmax())
        if p[i] < self.threshold:
            return "Uncategorised", float(p[i])
        return self.clf.classes_[i], float(p[i])


def _features(s):
    d = pd.DataFrame({"y": s.values}, index=s.index)
    d["t"] = np.arange(len(d))
    for l in (1, 2, 3):
        d[f"lag{l}"] = d.y.shift(l)
    d["roll3"] = d.y.shift(1).rolling(3).mean()
    m = d.index.month
    d["msin"] = np.sin(2 * np.pi * m / 12)
    d["mcos"] = np.cos(2 * np.pi * m / 12)
    return d.dropna()

FEATS = ["t", "lag1", "lag2", "lag3", "roll3", "msin", "mcos"]


def forecast_next(series, n=3):
    """Recursive multi-step forecast of a monthly expense series."""
    d = _features(series)
    model = LinearRegression().fit(d[FEATS], d.y)
    hist, out = list(series.values), []
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=n, freq="MS")
    for i, mo in enumerate(idx):
        ang = 2 * np.pi * mo.month / 12
        x = pd.DataFrame([[len(hist), hist[-1], hist[-2], hist[-3],
                           np.mean(hist[-3:]), np.sin(ang), np.cos(ang)]], columns=FEATS)
        p = max(float(model.predict(x)[0]), 0.0)
        out.append(p)
        hist.append(p)
    return pd.Series(out, index=idx)


def recommend_budget(income, forecast_by_cat, savings_rate=0.20):
    """Keep 'needs' intact, trim 'wants' so that spend <= income*(1-savings_rate)."""
    spendable = income * (1 - savings_rate)
    total = sum(forecast_by_cat.values())
    if total <= spendable:
        return {c: round(v) for c, v in forecast_by_cat.items()}
    needs = sum(v for c, v in forecast_by_cat.items() if c in NEEDS)
    wants = total - needs
    factor = max(spendable - needs, 0) / wants if wants > 0 else 0
    return {c: round(v if c in NEEDS else v * factor) for c, v in forecast_by_cat.items()}


def budget_status(spent, budget):
    r = spent / budget
    if r >= 1.0:
        return "EXCEEDED"
    if r >= 0.90:
        return "WARNING"
    if r >= 0.75:
        return "CAUTION"
    return "SAFE"


class AnomalyDetector:
    def __init__(self, contamination=0.03):
        self.model = IsolationForest(contamination=contamination, random_state=0)

    def fit(self, amounts, hours):
        self.model.fit(np.c_[amounts, hours])
        return self

    def is_anomaly(self, amount, hour):
        return bool(self.model.predict([[amount, hour]])[0] == -1)
