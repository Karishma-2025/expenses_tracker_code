"""Create a demo account with 30 months of synthetic transactions.

Usage:  python scripts/seed_demo.py
Login:  demo@example.com / demo1234
"""
import os
import sqlite3
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from werkzeug.security import generate_password_hash

from app import create_app
from ml.data_gen import generate_user_history

EMAIL, PASSWORD = "demo@example.com", "demo1234"


def main():
    app = create_app()
    path = app.config["DATABASE"]
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("DELETE FROM users WHERE email=?", (EMAIL,))
    cur = con.execute("INSERT INTO users(name, email, password_hash, monthly_income) VALUES (?,?,?,?)",
                      ("Demo User", EMAIL, generate_password_hash(PASSWORD), 40000))
    uid = cur.lastrowid
    ids = {r[1]: r[0] for r in con.execute("SELECT category_id, name FROM categories")}
    rows = generate_user_history(np.random.default_rng(2026), date.today(), months=30)
    con.executemany("""INSERT INTO transactions(user_id, category_id, amount, description, txn_date, txn_hour, payment_mode, confidence, is_anomaly)
                       VALUES (?,?,?,?,?,?,?,1.0,0)""",
                    [(uid, ids[r["category"]], r["amount"], r["description"], r["date"], r["hour"], r["mode"]) for r in rows])
    con.commit()
    con.close()
    client = app.test_client()
    client.post("/api/login", json={"email": EMAIL, "password": PASSWORD})
    res = client.post("/api/budget/recommend", json={"income": 40000, "savings_rate": 0.2, "save": True}).get_json()
    print(f"Demo user created with {len(rows)} transactions.")
    print(f"Recommended budget saved for {res['month']} (method: {res['method']}); total Rs. {res['total_budget']:,.0f}")
    print(f"Login with  {EMAIL}  /  {PASSWORD}")


if __name__ == "__main__":
    main()
