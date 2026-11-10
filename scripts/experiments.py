"""Re-run every experiment of the project report and print the result tables.

Usage:  python scripts/experiments.py [--charts DIR]
Numbers match Chapter 5 of the report (synthetic data, seed 42).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from ml.core import FEATS, _features, forecast_next
from ml.data_gen import generate_anomaly_data, generate_monthly, generate_transactions


def main(chart_dir=None):
    rng = np.random.default_rng(42)
    tx = generate_transactions(rng)
    monthly = generate_monthly(rng)
    amt, hr, y_true = generate_anomaly_data(rng)

    # ---- 1. categorisation ----
    Xtr, Xte, ytr, yte = train_test_split(tx.description, tx.category, test_size=0.2, random_state=7, stratify=tx.category)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    preds = {"Naive Bayes": MultinomialNB().fit(A, ytr).predict(B),
             "Logistic Regression": LogisticRegression(max_iter=1000).fit(A, ytr).predict(B)}
    print("\n=== Transaction categorisation (train %d / test %d) ===" % (len(Xtr), len(Xte)))
    for k, p in preds.items():
        print(f"{k:22s} accuracy={accuracy_score(yte, p)*100:6.2f}%   macro-F1={f1_score(yte, p, average='macro'):.4f}")

    # ---- 2. forecasting (last 12 months = test) ----
    d = _features(monthly["Total"])
    train, test = d.iloc[:-12], d.iloc[-12:]
    models = {"Moving Average (3m)": None, "Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=300, random_state=3),
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=3)}
    print("\n=== Forecasting total monthly expense (train %d rows / test %d rows) ===" % (len(train), len(test)))
    print(f"{'Model':22s}{'MAE':>9s}{'RMSE':>9s}{'MAPE%':>8s}{'R2':>8s}")
    fitted = {}
    for name, m in models.items():
        if m is None:
            p = test["roll3"].values
        else:
            m.fit(train[FEATS], train.y)
            p = m.predict(test[FEATS])
        fitted[name] = p
        print(f"{name:22s}{mean_absolute_error(test.y, p):9.0f}{mean_squared_error(test.y, p)**0.5:9.0f}"
              f"{np.mean(np.abs((test.y - p) / test.y))*100:8.2f}{r2_score(test.y, p):8.3f}")
    fc = forecast_next(monthly["Total"], 6)
    print("\n6-month forecast (Linear Regression):")
    for k, v in fc.items():
        print(f"  {k.strftime('%b %Y')}: Rs. {v:,.0f}")

    # ---- 3. anomaly detection ----
    iso = IsolationForest(contamination=0.03, random_state=0).fit(np.c_[amt, hr])
    flag = (iso.predict(np.c_[amt, hr]) == -1).astype(int)
    print("\n=== Anomaly detection (Isolation Forest) ===")
    print(f"precision={precision_score(y_true, flag):.2f} recall={recall_score(y_true, flag):.2f} F1={f1_score(y_true, flag):.2f}")

    if chart_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(chart_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(test.index, test.y, "k-o", label="Actual")
        for k, p in fitted.items():
            ax.plot(test.index, p, "--", label=k)
        ax.legend(fontsize=7); ax.set_title("Predicted vs actual (last 12 months)")
        fig.tight_layout(); fig.savefig(os.path.join(chart_dir, "forecast_test.png"), dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(monthly.index[-24:], monthly.Total[-24:], "k-", label="History")
        ax.plot(fc.index, fc.values, "r--o", label="Forecast"); ax.legend(); ax.set_title("6-month forecast")
        fig.tight_layout(); fig.savefig(os.path.join(chart_dir, "forecast_6m.png"), dpi=140); plt.close(fig)
        labels = sorted(tx.category.unique())
        cm = confusion_matrix(yte, preds["Logistic Regression"], labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        ax.set_title("Confusion matrix - Logistic Regression"); fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, "confusion_matrix.png"), dpi=140); plt.close(fig)
        print("\nCharts saved in", chart_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", default=None)
    main(ap.parse_args().charts)
