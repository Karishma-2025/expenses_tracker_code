"""Integration / system tests IT01-IT10 of the project report (Table 5.2) plus extra checks."""
import io
from datetime import date, timedelta

import pytest

from app import create_app

TODAY = date.today()


def add(client, amount, desc, d=None, **kw):
    return client.post("/api/expenses", json={"amount": amount, "description": desc, "date": (d or TODAY).isoformat(), **kw})


def past_month_date(k):
    d = TODAY.replace(day=15)
    for _ in range(k):
        d = (d.replace(day=1) - timedelta(days=1)).replace(day=15)
    return d


def fill_history(client, months=8):
    for k in range(1, months + 1):
        d = past_month_date(k)
        for desc, amt in [("zomato order", 3000 + 40 * k), ("bigbasket order", 5000), ("uber ride", 1800), ("amazon purchase", 2500 + 30 * k),
                          ("electricity bill", 2200), ("netflix subscription", 800), ("apollo pharmacy", 900), ("udemy course", 1200)]:
            assert add(client, amt, desc, d).status_code == 201


def test_it01_register_login_dashboard(client):
    assert client.get("/dashboard").status_code == 302                       # protected page
    r = client.post("/api/register", json={"name": "A", "email": "a@x.com", "password": "secret123", "monthly_income": 1000})
    assert r.status_code == 201
    client.post("/api/logout")
    assert client.post("/api/login", json={"email": "a@x.com", "password": "wrong"}).status_code == 401
    assert client.post("/api/login", json={"email": "a@x.com", "password": "secret123"}).status_code == 200
    assert client.get("/dashboard").status_code == 200


def test_it02_duplicate_email(user, app):
    r = app.test_client().post("/api/register", json={"name": "B", "email": "asha@example.com", "password": "secret123"})
    assert r.status_code == 409 and "already registered" in r.get_json()["error"]


def test_password_is_hashed(user, app):
    import sqlite3
    h = sqlite3.connect(app.config["DATABASE"]).execute("SELECT password_hash FROM users").fetchone()[0]
    assert "secret123" not in h and h.count("$") >= 2


def test_it03_add_expense_auto_category(user):
    r = add(user, 450, "zomato order")
    assert r.status_code == 201 and r.get_json()["category"] == "Food" and r.get_json()["anomaly"] is False
    s = user.get("/api/summary").get_json()
    assert s["total"] == 450 and [c for c in s["by_category"] if c["category"] == "Food"][0]["spent"] == 450


@pytest.mark.parametrize("amount,desc", [(0, "tea"), ("abc", "tea"), (-5, "tea"), (100, "   ")])
def test_it04_validation(user, amount, desc):
    r = add(user, amount, desc)
    assert r.status_code == 400 and "error" in r.get_json()
    assert user.get("/api/expenses").get_json() == []


def test_it05_csv_upload(user):
    rows = ["date,description,amount,time"] + [f"{TODAY.isoformat()},swiggy dinner upi,{100 + i},20:00" for i in range(97)]
    rows += [f"{TODAY.isoformat()},,50,10:00", f"{TODAY.isoformat()},uber ride,-5,10:00", "not-a-date,uber ride,100,10:00"]
    r = user.post("/api/upload", data={"file": (io.BytesIO("\n".join(rows).encode()), "s.csv")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["imported"] == 97 and j["skipped"] == 3 and {e["row"] for e in j["errors"]} == {99, 100, 101}


def test_upload_missing_columns(user):
    r = user.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2"), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "Missing" in r.get_json()["error"]


def test_it06_budget_warning_alert(user):
    assert user.put("/api/budget", json={"month": TODAY.strftime("%Y-%m"), "budgets": {"Shopping": 1000}}).status_code == 200
    r = add(user, 920, "amazon purchase")
    j = r.get_json()
    assert j["category"] == "Shopping" and j["budget_status"] == "WARNING"
    assert any(a["level"] == "WARNING" for a in j["alerts"])
    alerts = user.get("/api/alerts").get_json()
    assert any(a["level"] == "WARNING" for a in alerts)
    r2 = add(user, 200, "flipkart order").get_json()
    assert r2["budget_status"] == "EXCEEDED" and any(a["level"] == "EXCEEDED" for a in r2["alerts"])


def test_it07_forecast_needs_history(user):
    add(user, 300, "uber ride", past_month_date(1))
    j = user.get("/api/forecast?months=3").get_json()
    assert j["available"] is False and "months" in j["message"]


def test_forecast_and_budget_with_history(user):
    fill_history(user, 8)
    j = user.get("/api/forecast?months=6").get_json()
    assert j["available"] and len(j["total"]) == 6 and all(v >= 0 for v in j["total"]) and len(j["categories"]) == 8
    assert j["months"][0] == TODAY.strftime("%Y-%m")
    rec = user.post("/api/budget/recommend", json={"income": 20000, "savings_rate": 0.2, "save": True}).get_json()
    assert rec["total_budget"] <= rec["spendable"] + 8 and rec["saved"] is True
    saved = user.get("/api/budget").get_json()["budgets"]
    assert saved["Groceries"] > 0 and abs(sum(saved.values()) - rec["total_budget"]) < 1


def test_budget_recommend_validation(user):
    assert user.post("/api/budget/recommend", json={"income": 0}).status_code == 400
    assert user.post("/api/budget/recommend", json={"income": 5000, "savings_rate": 2}).status_code == 400


def test_it08_data_isolation(user, app):
    tid = add(user, 450, "zomato order").get_json()["id"]
    b = app.test_client()
    b.post("/api/register", json={"name": "B", "email": "b@example.com", "password": "secret123"})
    assert b.put(f"/api/expenses/{tid}", json={"category": "Bills"}).status_code == 403
    assert b.delete(f"/api/expenses/{tid}").status_code == 403
    assert b.get("/api/expenses").get_json() == []
    assert b.get("/api/summary").get_json()["total"] == 0
    assert b.put(f"/api/expenses/99999", json={"category": "Bills"}).status_code == 404


def test_login_required_for_api(client):
    for path in ["/api/expenses", "/api/summary", "/api/forecast", "/api/alerts", "/api/export"]:
        assert client.get(path).status_code == 401


def test_it09_correct_category_and_retrain(user):
    tid = add(user, 999, "mystery shop xyz").get_json()["id"]
    r = user.put(f"/api/expenses/{tid}", json={"category": "Health"})
    assert r.status_code == 200 and r.get_json()["category"] == "Health"
    assert user.put(f"/api/expenses/{tid}", json={"category": "Nonsense"}).status_code == 400
    assert user.post("/api/retrain").get_json()["retrained_with"] == 1
    assert user.get("/api/predict?description=mystery shop xyz").get_json()["category"] == "Health"


def test_retrain_without_corrections(user):
    assert user.post("/api/retrain").status_code == 400


def test_it10_restart_keeps_data_and_models(user, app, model_dir):
    add(user, 450, "zomato order")
    app2 = create_app({"TESTING": True, "DATABASE": app.config["DATABASE"], "MODEL_DIR": model_dir, "SECRET_KEY": "test"})
    c2 = app2.test_client()
    assert c2.post("/api/login", json={"email": "asha@example.com", "password": "secret123"}).status_code == 200
    assert len(c2.get("/api/expenses").get_json()) == 1
    assert add(c2, 120, "uber ride").get_json()["category"] == "Transport"


def test_anomaly_flagged_and_alert_created(user):
    r = add(user, 150000, "amazon purchase", time="03:12").get_json()
    assert r["anomaly"] is True and any(a["level"] == "ANOMALY" for a in r["alerts"])
    aid = [a for a in user.get("/api/alerts").get_json() if a["level"] == "ANOMALY"][0]["id"]
    assert user.put(f"/api/alerts/{aid}/review").status_code == 200
    assert [a for a in user.get("/api/alerts").get_json() if a["id"] == aid][0]["read"] is True
    assert add(user, 250, "chai tapri", time="14:00").get_json()["anomaly"] is False


def test_manual_category_override(user):
    assert add(user, 300, "something odd", category="Education").get_json()["category"] == "Education"
    assert add(user, 300, "x", category="Bogus").status_code == 400


def test_export_and_delete_account(user):
    add(user, 450, "zomato order")
    ex = user.get("/api/export").get_json()
    assert ex["user"]["email"] == "asha@example.com" and len(ex["transactions"]) == 1
    assert user.delete("/api/account", json={"password": "wrong"}).status_code == 403
    assert user.delete("/api/account", json={"password": "secret123"}).status_code == 200
    assert user.get("/api/me").status_code == 401
    assert user.post("/api/login", json={"email": "asha@example.com", "password": "secret123"}).status_code == 401


def test_update_income_and_summary(user):
    assert user.put("/api/me", json={"monthly_income": 55000}).status_code == 200
    assert user.get("/api/summary").get_json()["income"] == 55000
    assert user.put("/api/me", json={"monthly_income": "abc"}).status_code == 400
