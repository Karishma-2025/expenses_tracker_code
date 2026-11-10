"""Unit tests TC01-TC15 of the project report (Table 5.1) for ml/core.py."""
import numpy as np
import pandas as pd
import pytest

from ml.core import AnomalyDetector, Categoriser, budget_status, forecast_next, recommend_budget, validate_transaction
from ml.data_gen import generate_anomaly_data, generate_monthly, generate_transactions


@pytest.fixture(scope="module")
def cat():
    tx = generate_transactions(np.random.default_rng(42))
    return Categoriser().fit(tx.description, tx.category)


@pytest.fixture(scope="module")
def monthly():
    rng = np.random.default_rng(42)
    generate_transactions(rng)
    return generate_monthly(rng)


def test_tc01_food(cat): assert cat.predict("swiggy dinner upi")[0] == "Food"
def test_tc02_transport(cat): assert cat.predict("UBER RIDE 4582")[0] == "Transport"
def test_tc03_entertainment(cat): assert cat.predict("netflix subscription")[0] == "Entertainment"
def test_tc04_bills(cat): assert cat.predict("electricity bill")[0] == "Bills"
def test_tc05_numeric_text_is_uncategorised(cat): assert cat.predict("1234")[0] == "Uncategorised"


def test_tc06_negative_amount_rejected():
    with pytest.raises(ValueError):
        validate_transaction(-50, "tea")


def test_tc06b_empty_description_rejected():
    with pytest.raises(ValueError):
        validate_transaction(50, "   ")


def test_tc07_forecast_length(monthly): assert len(forecast_next(monthly["Total"], 6)) == 6


def test_tc08_forecast_range(monthly):
    fc = forecast_next(monthly["Total"], 6)
    assert (fc > 0).all() and (fc < 2 * monthly.Total.max()).all()


FORECAST = {"Groceries": 9000, "Bills": 6000, "Transport": 4000, "Health": 2000, "Education": 2000,
            "Food": 9000, "Shopping": 8000, "Entertainment": 5000}


def test_tc09_budget_fits_spendable():
    assert sum(recommend_budget(50000, FORECAST).values()) <= 40000 + 8


def test_tc10_needs_preserved():
    assert recommend_budget(50000, FORECAST)["Groceries"] == 9000


def test_budget_unchanged_when_forecast_fits():
    assert recommend_budget(100000, FORECAST) == {c: round(v) for c, v in FORECAST.items()}


def test_tc11_warning(): assert budget_status(950, 1000) == "WARNING"
def test_tc12_exceeded(): assert budget_status(1200, 1000) == "EXCEEDED"
def test_tc13_safe(): assert budget_status(400, 1000) == "SAFE"
def test_status_caution_boundary(): assert budget_status(750, 1000) == "CAUTION" and budget_status(749, 1000) == "SAFE"


@pytest.fixture(scope="module")
def detector():
    rng = np.random.default_rng(5)
    amt, hr = rng.lognormal(5.5, 0.9, 970), rng.integers(8, 23, 970)
    return AnomalyDetector().fit(np.r_[amt, rng.lognormal(9, 0.4, 30)], np.r_[hr, rng.integers(0, 5, 30)])


def test_tc14_large_night_flagged(detector): assert detector.is_anomaly(150000, 3)
def test_tc15_normal_not_flagged(detector): assert not detector.is_anomaly(250, 14)


def test_anomaly_detector_f1_on_synthetic_data():
    amt, hr, y = generate_anomaly_data(np.random.default_rng(7))
    det = AnomalyDetector().fit(amt, hr)
    pred = det.model.predict(np.c_[amt, hr]) == -1
    tp = (pred & (y == 1)).sum()
    assert tp / max(pred.sum(), 1) > 0.6 and tp / 30 > 0.6
