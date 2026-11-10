"""Synthetic data generators (real bank data is private).

The generators reproduce the datasets used in the project report. To reproduce the
report numbers exactly, call them in this order with ONE shared numpy Generator:
transactions -> monthly series -> anomaly data.
"""
import numpy as np
import pandas as pd

CATEGORY_MERCHANTS = {
    "Food": ["zomato order", "swiggy dinner", "domino's pizza", "mcdonald's", "cafe coffee day", "starbucks coffee",
             "local restaurant", "biryani house", "chai tapri", "dominos", "kfc bucket", "subway sandwich",
             "lunch canteen", "burger king"],
    "Groceries": ["bigbasket order", "dmart groceries", "blinkit items", "zepto order", "reliance fresh",
                  "vegetable market", "kirana store", "milk dairy", "more supermarket", "grofers"],
    "Transport": ["uber ride", "ola cab", "metro card recharge", "petrol pump", "indian oil fuel", "rapido bike",
                  "irctc train ticket", "bus pass", "auto rickshaw", "fastag recharge"],
    "Shopping": ["amazon purchase", "flipkart order", "myntra clothes", "ajio fashion", "nykaa beauty",
                 "decathlon sports", "croma electronics", "zara store", "h&m store", "lifestyle mall"],
    "Bills": ["electricity bill", "jio recharge", "airtel postpaid", "broadband bill", "water bill",
              "gas cylinder booking", "dth recharge", "mobile recharge", "maintenance charges", "bsnl bill"],
    "Entertainment": ["netflix subscription", "bookmyshow movie", "spotify premium", "hotstar plan", "pvr cinemas",
                      "steam game", "youtube premium", "amazon prime", "gaming zone", "concert ticket"],
    "Health": ["apollo pharmacy", "medplus medicines", "doctor consultation", "lab test", "gym membership",
               "1mg order", "dental clinic", "hospital bill", "eye checkup", "practo appointment"],
    "Education": ["udemy course", "coursera fee", "college fee", "book store", "stationery shop", "exam fee",
                  "coaching class", "library fine", "online tutorial", "coding bootcamp"],
}
CATEGORIES = list(CATEGORY_MERCHANTS)
SUFFIX = ["", "upi", "pos", "txn", "payment", "ref", "online"]
BASE = {"Food": 5500, "Groceries": 6000, "Transport": 2800, "Shopping": 3500,
        "Bills": 3800, "Entertainment": 1500, "Health": 1200, "Education": 1800}
PAYMENT_MODES = ["UPI", "Card", "Cash", "Net Banking"]


def generate_transactions(rng, n_per_cat=300):
    """Labelled (description, category) pairs; ~15% of descriptions are corrupted."""
    rows = []
    for cat, merchants in CATEGORY_MERCHANTS.items():
        for _ in range(n_per_cat):
            m = rng.choice(merchants)
            words = m.split()
            if rng.random() < 0.15:  # imitate truncated / noisy bank narration
                words[rng.integers(len(words))] = words[rng.integers(len(words))][::-1][:3] + "x"
            text = " ".join(words) + " " + rng.choice(SUFFIX)
            rows.append((text.strip(), cat))
    return pd.DataFrame(rows, columns=["description", "category"]).sample(frac=1, random_state=1).reset_index(drop=True)


def _season(cat, base, month):
    month = np.asarray(month)
    if cat == "Shopping":
        return base * 0.5 * ((month == 10) | (month == 11)) + base * 0.25 * (month == 12)
    if cat == "Food":
        return base * 0.15 * ((month == 12) | (month == 5))
    if cat == "Entertainment":
        return base * 0.3 * ((month == 12) | (month == 6))
    if cat == "Bills":
        return base * 0.15 * ((month >= 4) & (month <= 6))
    if cat == "Education":
        return base * 1.2 * ((month == 7) | (month == 1))
    if cat == "Health":
        return base * 0.3 * ((month == 2) | (month == 9))
    return 0


def generate_monthly(rng, start="2021-01-01", periods=60):
    """Monthly spending per category: trend + seasonality + noise."""
    months = pd.date_range(start, periods=periods, freq="MS")
    data = {}
    for cat, base in BASE.items():
        t = np.arange(periods)
        trend = base * (1 + 0.006 * t)
        season = _season(cat, base, months.month.values)
        noise = rng.normal(0, base * 0.08, periods)
        data[cat] = np.round(trend + season + noise, 0)
    df = pd.DataFrame(data, index=months)
    df["Total"] = df.sum(axis=1)
    return df


def generate_anomaly_data(rng, n=1000, n_anom=30):
    """Amount + hour of day; the last n_anom rows are anomalies (big amount at night)."""
    amt = np.concatenate([rng.lognormal(5.5, 0.9, n - n_anom), rng.lognormal(9, 0.4, n_anom)])
    hr = np.concatenate([rng.integers(8, 23, n - n_anom), rng.integers(0, 5, n_anom)])
    y = np.r_[np.zeros(n - n_anom), np.ones(n_anom)]
    return amt, hr, y


def generate_user_history(rng, end_date, months=30):
    """Realistic transaction list for the demo user (several purchases per category per month)."""
    end = pd.Timestamp(end_date).normalize()
    first = (end.replace(day=1) - pd.DateOffset(months=months - 1))
    idx = pd.date_range(first, end.replace(day=1), freq="MS")
    out = []
    for k, ms in enumerate(idx):
        month_end = (ms + pd.offsets.MonthEnd(0)).day
        is_cur = ms.year == end.year and ms.month == end.month
        last_day = end.day if is_cur else month_end
        frac = last_day / month_end
        for cat, base in BASE.items():
            target = (base * (1 + 0.006 * k) + _season(cat, base, ms.month) + rng.normal(0, base * 0.08)) * frac
            n_tx = int(rng.integers(3, 8))
            parts = rng.dirichlet(np.ones(n_tx)) * max(float(target), 100.0)
            for p in parts:
                desc = rng.choice(CATEGORY_MERCHANTS[cat]) + " " + rng.choice(SUFFIX)
                day = ms + pd.Timedelta(days=int(rng.integers(0, last_day)))
                out.append({"date": day.strftime("%Y-%m-%d"), "hour": int(rng.integers(8, 23)),
                            "description": desc.strip(), "amount": round(float(max(p, 20)), 2),
                            "category": cat, "mode": str(rng.choice(PAYMENT_MODES))})
    return out
