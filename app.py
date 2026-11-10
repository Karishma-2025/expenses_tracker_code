"""AI Based Expense Tracker and Budget Prediction - Flask application.

Run:  python app.py        (then open http://127.0.0.1:5000)
"""
import csv
import io
import os
import re
import sqlite3
from datetime import date, datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from ml.core import AnomalyDetector, NEEDS, budget_status, forecast_next, recommend_budget, validate_transaction
from ml.data_gen import CATEGORIES
from ml.registry import ModelRegistry

CAT_TYPE = {c: ("Need" if c in NEEDS else "Want") for c in CATEGORIES}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
LEVEL_TEXT = {"CAUTION": "75%", "WARNING": "90%", "EXCEEDED": "100%"}


# ------------------------------------------------------------------ database helpers
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(g.app_config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def init_db(path):
    con = sqlite3.connect(path)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql"), encoding="utf8") as f:
        con.executescript(f.read())
    for i, c in enumerate(CATEGORIES, 1):
        con.execute("INSERT OR IGNORE INTO categories(category_id, name, type) VALUES (?,?,?)", (i, c, CAT_TYPE[c]))
    con.commit()
    con.close()


def cat_ids(db):
    return {r["name"]: r["category_id"] for r in db.execute("SELECT category_id, name FROM categories")}


# ------------------------------------------------------------------ small utilities
def current_month():
    return date.today().strftime("%Y-%m")


def valid_month(s):
    if s and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s):
        return s
    return current_month()


def parse_date(s):
    s = str(s or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError("Date must be in YYYY-MM-DD format")


def parse_hour(time_str=None, hour=None):
    if hour not in (None, ""):
        h = int(hour)
    elif time_str:
        h = int(str(time_str).split(":")[0])
    else:
        return 12
    if not 0 <= h <= 23:
        raise ValueError("Hour must be between 0 and 23")
    return h


def parse_amount(v):
    try:
        a = float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        raise ValueError("Amount must be a number")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("Amount must be greater than zero")
    if a > 1e8:
        raise ValueError("Amount is too large")
    return round(a, 2)


def month_add(m, k):
    y, mo = map(int, m.split("-"))
    t = y * 12 + (mo - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify(error="Login required"), 401
            return redirect(url_for("login_page"))
        return fn(*a, **kw)
    return wrapper


# ------------------------------------------------------------------ factory
def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"]
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)
    init_db(app.config["DATABASE"])
    registry = ModelRegistry(app.config["MODEL_DIR"], threshold=app.config["CATEGORY_CONFIDENCE"])
    app.extensions["registry"] = registry
    detector_cache = {}

    @app.before_request
    def _bind():
        g.app_config = app.config

    @app.teardown_appcontext
    def _close(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ---------------------------------------------------------------- domain helpers
    def detector_for(db, uid):
        """Personal Isolation Forest once the user has >= 50 transactions, otherwise the global one."""
        n = db.execute("SELECT COUNT(*) FROM transactions WHERE user_id=?", (uid,)).fetchone()[0]
        if n < 50:
            return registry.detector
        key = (uid, n // 25)
        if key not in detector_cache:
            rows = db.execute("SELECT amount, txn_hour FROM transactions WHERE user_id=?", (uid,)).fetchall()
            detector_cache.clear()
            detector_cache[key] = AnomalyDetector().fit([r[0] for r in rows], [r[1] for r in rows])
        return detector_cache[key]

    def spent_by_category(db, uid, month):
        rows = db.execute("""SELECT COALESCE(c.name,'Uncategorised') AS cat, SUM(t.amount) AS s FROM transactions t
                             LEFT JOIN categories c ON c.category_id=t.category_id
                             WHERE t.user_id=? AND strftime('%Y-%m', t.txn_date)=? GROUP BY cat""", (uid, month)).fetchall()
        return {r["cat"]: r["s"] for r in rows}

    def budgets_for(db, uid, month):
        rows = db.execute("""SELECT c.name, b.limit_amount FROM budgets b JOIN categories c ON c.category_id=b.category_id
                             WHERE b.user_id=? AND b.month=?""", (uid, month)).fetchall()
        return {r["name"]: r["limit_amount"] for r in rows}

    def add_alert(db, uid, level, message, ref_key):
        cur = db.execute("INSERT OR IGNORE INTO alerts(user_id, message, level, ref_key) VALUES (?,?,?,?)", (uid, message, level, ref_key))
        return {"level": level, "message": message} if cur.rowcount else None

    def evaluate_alerts(db, uid, month, only_cat=None):
        """Create CAUTION / WARNING / EXCEEDED alerts (once per level per category per month)."""
        spent, bud, new = spent_by_category(db, uid, month), budgets_for(db, uid, month), []
        order = ["CAUTION", "WARNING", "EXCEEDED"]
        for cat, limit in bud.items():
            if only_cat and cat != only_cat or limit <= 0:
                continue
            st = budget_status(spent.get(cat, 0.0), limit)
            if st == "SAFE":
                continue
            for lv in [st]:
                pct = spent.get(cat, 0.0) / limit * 100
                a = add_alert(db, uid, lv, f"{cat} budget {pct:.0f}% used (Rs. {spent.get(cat, 0):,.0f} of Rs. {limit:,.0f}) for {month}",
                              f"{month}:{cat}:{lv}")
                if a:
                    new.append(a)
        return new

    def insert_transaction(db, uid, amount, description, d, hour, mode, category=None, detector=None):
        validate_transaction(amount, description)
        ids = cat_ids(db)
        if category:
            if category not in ids:
                raise ValueError(f"Unknown category '{category}'")
            conf = 1.0
        else:
            category, conf = registry.categoriser_for(uid).predict(description)
        flagged = (detector or detector_for(db, uid)).is_anomaly(amount, hour)
        cur = db.execute("""INSERT INTO transactions(user_id, category_id, amount, description, txn_date, txn_hour, payment_mode, confidence, is_anomaly)
                            VALUES (?,?,?,?,?,?,?,?,?)""", (uid, ids.get(category), amount, description.strip(), d.isoformat(), hour, mode, conf, int(flagged)))
        alerts = []
        if flagged:
            a = add_alert(db, uid, "ANOMALY", f"Rs. {amount:,.0f} for '{description.strip()}' at {hour:02d}:00 on {d.isoformat()} is unusual for your pattern. Was this you?",
                          f"anomaly:{cur.lastrowid}")
            alerts.append(a) if a else None
        alerts += evaluate_alerts(db, uid, d.strftime("%Y-%m"), category if category in ids else None)
        return {"id": cur.lastrowid, "category": category, "confidence": round(conf, 2), "anomaly": bool(flagged), "alerts": alerts}

    def monthly_frame(db, uid):
        """Complete months only (current month excluded), zero-filled, columns = categories."""
        rows = db.execute("""SELECT strftime('%Y-%m', t.txn_date) AS m, COALESCE(c.name,'Uncategorised') AS cat, SUM(t.amount) AS s
                             FROM transactions t LEFT JOIN categories c ON c.category_id=t.category_id
                             WHERE t.user_id=? AND strftime('%Y-%m', t.txn_date) < ? GROUP BY m, cat""", (uid, current_month())).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["m", "cat", "s"]).pivot_table(index="m", columns="cat", values="s", aggfunc="sum").fillna(0.0)
        idx = pd.date_range(df.index.min() + "-01", df.index.max() + "-01", freq="MS")
        df.index = pd.to_datetime(df.index + "-01")
        df = df.reindex(idx, fill_value=0.0)
        for c in CATEGORIES:
            if c not in df.columns:
                df[c] = 0.0
        return df[CATEGORIES]

    def forecast_all(db, uid, n):
        df = monthly_frame(db, uid)
        need = app.config["MIN_FORECAST_MONTHS"]
        if df is None or len(df) < need:
            have = 0 if df is None else len(df)
            return None, f"At least {need} complete months of data are needed for a forecast (you have {have}). Add or upload more transactions."
        res = {c: forecast_next(df[c], n) for c in CATEGORIES}
        months = [d.strftime("%Y-%m") for d in res[CATEGORIES[0]].index]
        cats = {c: [round(float(v), 2) for v in res[c].values] for c in CATEGORIES}
        total = [round(sum(cats[c][i] for c in CATEGORIES), 2) for i in range(n)]
        ids = cat_ids(db)
        for c in CATEGORIES:
            for m, v in zip(months, cats[c]):
                db.execute("""INSERT INTO predictions(user_id, category_id, target_month, predicted_amt, model_name) VALUES (?,?,?,?,?)
                              ON CONFLICT(user_id, category_id, target_month) DO UPDATE SET predicted_amt=excluded.predicted_amt, model_name=excluded.model_name""",
                           (uid, ids[c], m, v, "Linear Regression"))
        hist = df.sum(axis=1).tail(12)
        return {"available": True, "model": "Linear Regression", "months": months, "categories": cats, "total": total,
                "history": {"months": [d.strftime("%Y-%m") for d in hist.index], "total": [round(float(v), 2) for v in hist.values]}}, None

    def user_row(db):
        return db.execute("SELECT * FROM users WHERE user_id=?", (session["user_id"],)).fetchone()

    def tx_json(r):
        return {"id": r["txn_id"], "date": r["txn_date"], "description": r["description"], "amount": r["amount"],
                "category": r["cat"] or "Uncategorised", "mode": r["payment_mode"], "confidence": r["confidence"],
                "anomaly": bool(r["is_anomaly"]), "hour": r["txn_hour"]}

    TX_SQL = """SELECT t.*, c.name AS cat FROM transactions t LEFT JOIN categories c ON c.category_id=t.category_id"""

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return redirect(url_for("dashboard_page" if "user_id" in session else "login_page"))

    @app.get("/login")
    def login_page():
        return render_template("login.html")

    @app.get("/register")
    def register_page():
        return render_template("register.html")

    @app.get("/logout")
    def logout_page():
        session.clear()
        return redirect(url_for("login_page"))

    def make_page(path, endpoint, template, title):
        @app.get(path, endpoint=endpoint)
        @login_required
        def _page():
            return render_template(template, title=title, user_name=user_row(get_db())["name"], categories=CATEGORIES, page=endpoint)
        return _page

    for p, e, t, ti in [("/dashboard", "dashboard_page", "dashboard.html", "Dashboard"), ("/add", "add_page", "add_expense.html", "Add Expense"),
                        ("/transactions", "transactions_page", "transactions.html", "Transactions"), ("/budget", "budget_page", "budget.html", "Budget Planner"),
                        ("/forecast", "forecast_page", "forecast.html", "Forecast"), ("/alerts", "alerts_page", "alerts.html", "Alerts")]:
        make_page(p, e, t, ti)

    # ---------------------------------------------------------------- auth API
    @app.post("/api/register")
    def api_register():
        d = request.get_json(silent=True) or {}
        name, email, pw = str(d.get("name", "")).strip(), str(d.get("email", "")).strip().lower(), str(d.get("password", ""))
        if not name:
            return jsonify(error="Name is required"), 400
        if not EMAIL_RE.match(email):
            return jsonify(error="Enter a valid e-mail address"), 400
        if len(pw) < 6:
            return jsonify(error="Password must have at least 6 characters"), 400
        try:
            income = float(d.get("monthly_income") or 0)
            if income < 0:
                raise ValueError
        except ValueError:
            return jsonify(error="Monthly income must be a positive number"), 400
        db = get_db()
        if db.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            return jsonify(error="E-mail already registered"), 409
        cur = db.execute("INSERT INTO users(name, email, password_hash, monthly_income) VALUES (?,?,?,?)", (name, email, generate_password_hash(pw), income))
        db.commit()
        session.clear(); session["user_id"] = cur.lastrowid
        return jsonify(id=cur.lastrowid, name=name, email=email), 201

    @app.post("/api/login")
    def api_login():
        d = request.get_json(silent=True) or {}
        db = get_db()
        u = db.execute("SELECT * FROM users WHERE email=?", (str(d.get("email", "")).strip().lower(),)).fetchone()
        if not u or not check_password_hash(u["password_hash"], str(d.get("password", ""))):
            return jsonify(error="Invalid e-mail or password"), 401
        session.clear(); session["user_id"] = u["user_id"]
        return jsonify(id=u["user_id"], name=u["name"], email=u["email"])

    @app.post("/api/logout")
    def api_logout():
        session.clear()
        return jsonify(ok=True)

    @app.get("/api/me")
    @login_required
    def api_me():
        u = user_row(get_db())
        return jsonify(id=u["user_id"], name=u["name"], email=u["email"], monthly_income=u["monthly_income"])

    @app.put("/api/me")
    @login_required
    def api_me_update():
        d = request.get_json(silent=True) or {}
        try:
            income = float(d.get("monthly_income"))
            if income < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify(error="Monthly income must be a positive number"), 400
        db = get_db()
        db.execute("UPDATE users SET monthly_income=? WHERE user_id=?", (income, session["user_id"]))
        db.commit()
        return jsonify(monthly_income=income)

    @app.get("/api/categories")
    @login_required
    def api_categories():
        return jsonify([{"name": c, "type": CAT_TYPE[c]} for c in CATEGORIES])

    # ---------------------------------------------------------------- transactions API
    @app.post("/api/expenses")
    @login_required
    def api_add_expense():
        d = request.get_json(silent=True) or {}
        db, uid = get_db(), session["user_id"]
        try:
            amount = parse_amount(d.get("amount"))
            desc = str(d.get("description", "")).strip()
            validate_transaction(amount, desc)
            dt = parse_date(d.get("date") or date.today().isoformat())
            hour = parse_hour(d.get("time"), d.get("hour"))
            res = insert_transaction(db, uid, amount, desc, dt, hour, d.get("payment_mode") or d.get("mode"), d.get("category") or None)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        db.commit()
        month = dt.strftime("%Y-%m")
        bud = budgets_for(db, uid, month)
        res["budget_status"] = budget_status(spent_by_category(db, uid, month).get(res["category"], 0.0), bud[res["category"]]) if res["category"] in bud and bud[res["category"]] > 0 else "NO BUDGET"
        return jsonify(res), 201

    @app.get("/api/expenses")
    @login_required
    def api_list_expenses():
        db, uid = get_db(), session["user_id"]
        month = request.args.get("month")
        q, args = TX_SQL + " WHERE t.user_id=?", [uid]
        if month:
            q += " AND strftime('%Y-%m', t.txn_date)=?"; args.append(valid_month(month))
        if request.args.get("category"):
            q += " AND c.name=?"; args.append(request.args["category"])
        q += " ORDER BY t.txn_date DESC, t.txn_id DESC LIMIT ?"
        args.append(min(int(request.args.get("limit", 500)), 2000))
        return jsonify([tx_json(r) for r in db.execute(q, args)])

    def owned_tx(db, tid):
        r = db.execute(TX_SQL + " WHERE t.txn_id=?", (tid,)).fetchone()
        if r is None:
            return None, (jsonify(error="Transaction not found"), 404)
        if r["user_id"] != session["user_id"]:
            return None, (jsonify(error="Forbidden"), 403)
        return r, None

    @app.put("/api/expenses/<int:tid>")
    @login_required
    def api_update_expense(tid):
        db, uid = get_db(), session["user_id"]
        r, err = owned_tx(db, tid)
        if err:
            return err
        d = request.get_json(silent=True) or {}
        try:
            amount = parse_amount(d["amount"]) if "amount" in d else r["amount"]
            desc = str(d.get("description", r["description"])).strip()
            validate_transaction(amount, desc)
            dt = parse_date(d["date"]) if "date" in d else parse_date(r["txn_date"])
            cid = r["category_id"]
            if "category" in d:
                ids = cat_ids(db)
                if d["category"] not in ids:
                    return jsonify(error="Unknown category"), 400
                cid = ids[d["category"]]
                db.execute("INSERT INTO corrections(user_id, description, category) VALUES (?,?,?) ON CONFLICT(user_id, description) DO UPDATE SET category=excluded.category",
                           (uid, desc.lower(), d["category"]))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        db.execute("UPDATE transactions SET amount=?, description=?, txn_date=?, category_id=?, confidence=? WHERE txn_id=?",
                   (amount, desc, dt.isoformat(), cid, 1.0 if "category" in d else r["confidence"], tid))
        db.commit()
        return jsonify(tx_json(db.execute(TX_SQL + " WHERE t.txn_id=?", (tid,)).fetchone()))

    @app.delete("/api/expenses/<int:tid>")
    @login_required
    def api_delete_expense(tid):
        db = get_db()
        r, err = owned_tx(db, tid)
        if err:
            return err
        db.execute("DELETE FROM transactions WHERE txn_id=?", (tid,))
        db.commit()
        return jsonify(deleted=tid)

    @app.post("/api/upload")
    @login_required
    def api_upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify(error="Choose a CSV file"), 400
        try:
            text = f.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            return jsonify(error="File must be UTF-8 encoded CSV"), 400
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return jsonify(error="Empty file"), 400
        cols = {c.strip().lower(): c for c in reader.fieldnames if c}
        missing = [c for c in ("date", "description", "amount") if c not in cols]
        if missing:
            return jsonify(error="Missing column(s): " + ", ".join(missing)), 400
        db, uid = get_db(), session["user_id"]
        det, imported, errors = detector_for(db, uid), 0, []
        for i, row in enumerate(reader, start=2):
            try:
                g_ = lambda k: (row.get(cols[k]) if k in cols else None)
                amount = parse_amount(g_("amount"))
                desc = str(g_("description") or "").strip()
                validate_transaction(amount, desc)
                cat = (g_("category") or "").strip() or None
                insert_transaction(db, uid, amount, desc, parse_date(g_("date")), parse_hour(g_("time")), g_("mode") or g_("payment_mode"), cat, det)
                imported += 1
            except ValueError as e:
                errors.append({"row": i, "error": str(e)})
        db.commit()
        return jsonify(imported=imported, skipped=len(errors), errors=errors[:50])

    # ---------------------------------------------------------------- summary / forecast / budget / alerts
    @app.get("/api/summary")
    @login_required
    def api_summary():
        db, uid = get_db(), session["user_id"]
        month = valid_month(request.args.get("month"))
        spent, bud = spent_by_category(db, uid, month), budgets_for(db, uid, month)
        rows = []
        for c in CATEGORIES + (["Uncategorised"] if "Uncategorised" in spent else []):
            s, b = round(spent.get(c, 0.0), 2), bud.get(c)
            rows.append({"category": c, "type": CAT_TYPE.get(c, "-"), "spent": s, "budget": b,
                         "percent": round(s / b * 100, 1) if b else None, "status": budget_status(s, b) if b else "NO BUDGET"})
        u = user_row(db)
        unread = db.execute("SELECT COUNT(*) FROM alerts WHERE user_id=? AND is_read=0", (uid,)).fetchone()[0]
        return jsonify(month=month, total=round(sum(spent.values()), 2), income=u["monthly_income"], budget_total=round(sum(bud.values()), 2),
                       by_category=rows, unread_alerts=unread)

    @app.get("/api/forecast")
    @login_required
    def api_forecast():
        db, uid = get_db(), session["user_id"]
        n = max(1, min(int(request.args.get("months", 3) or 3), 6))
        res, msg = forecast_all(db, uid, n)
        db.commit()
        if res is None:
            return jsonify(available=False, message=msg)
        return jsonify(res)

    @app.post("/api/budget/recommend")
    @login_required
    def api_budget_recommend():
        d = request.get_json(silent=True) or {}
        db, uid = get_db(), session["user_id"]
        try:
            income = float(d.get("income") or user_row(db)["monthly_income"] or 0)
            rate = float(d.get("savings_rate", 0.2))
        except ValueError:
            return jsonify(error="Income and savings rate must be numbers"), 400
        if income <= 0:
            return jsonify(error="Enter a monthly income greater than zero"), 400
        if not 0 <= rate < 1:
            return jsonify(error="Savings rate must be between 0 and 0.99"), 400
        month = current_month()
        res, _ = forecast_all(db, uid, 1)
        if res is not None:
            fc, method = {c: res["categories"][c][0] for c in CATEGORIES}, "Linear Regression forecast"
        else:
            df = monthly_frame(db, uid)
            if df is None or df.empty:
                return jsonify(error="Add some transactions first so the system can learn your spending pattern"), 400
            fc, method = {c: float(df[c].mean()) for c in CATEGORIES}, "average of available months (less than 6 months of history)"
        budget = recommend_budget(income, fc, rate)
        if d.get("save"):
            ids = cat_ids(db)
            for c, v in budget.items():
                db.execute("""INSERT INTO budgets(user_id, category_id, month, limit_amount) VALUES (?,?,?,?)
                              ON CONFLICT(user_id, category_id, month) DO UPDATE SET limit_amount=excluded.limit_amount""", (uid, ids[c], month, v))
            evaluate_alerts(db, uid, month)
        db.commit()
        return jsonify(month=month, method=method, income=income, savings_rate=rate, spendable=round(income * (1 - rate), 2),
                       total_forecast=round(sum(fc.values()), 2), total_budget=round(sum(budget.values()), 2),
                       categories=[{"category": c, "type": CAT_TYPE[c], "forecast": round(fc[c]), "budget": budget[c]} for c in CATEGORIES],
                       saved=bool(d.get("save")))

    @app.get("/api/budget")
    @login_required
    def api_budget_get():
        month = valid_month(request.args.get("month"))
        return jsonify(month=month, budgets=budgets_for(get_db(), session["user_id"], month))

    @app.put("/api/budget")
    @login_required
    def api_budget_put():
        d = request.get_json(silent=True) or {}
        month, db, uid, ids = valid_month(d.get("month")), get_db(), session["user_id"], None
        ids = cat_ids(db)
        try:
            for c, v in (d.get("budgets") or {}).items():
                if c not in ids:
                    raise ValueError(f"Unknown category '{c}'")
                v = float(v)
                if v < 0:
                    raise ValueError("Budget cannot be negative")
                db.execute("""INSERT INTO budgets(user_id, category_id, month, limit_amount) VALUES (?,?,?,?)
                              ON CONFLICT(user_id, category_id, month) DO UPDATE SET limit_amount=excluded.limit_amount""", (uid, ids[c], month, v))
        except (ValueError, TypeError) as e:
            db.rollback()
            return jsonify(error=str(e)), 400
        evaluate_alerts(db, uid, month)
        db.commit()
        return jsonify(month=month, budgets=budgets_for(db, uid, month))

    @app.get("/api/alerts")
    @login_required
    def api_alerts():
        db, uid = get_db(), session["user_id"]
        rows = db.execute("SELECT * FROM alerts WHERE user_id=? ORDER BY alert_id DESC LIMIT 100", (uid,)).fetchall()
        return jsonify([{"id": r["alert_id"], "level": r["level"], "message": r["message"], "read": bool(r["is_read"]), "created_at": r["created_at"]} for r in rows])

    @app.put("/api/alerts/<int:aid>/review")
    @login_required
    def api_alert_review(aid):
        db = get_db()
        r = db.execute("SELECT user_id FROM alerts WHERE alert_id=?", (aid,)).fetchone()
        if r is None:
            return jsonify(error="Alert not found"), 404
        if r["user_id"] != session["user_id"]:
            return jsonify(error="Forbidden"), 403
        db.execute("UPDATE alerts SET is_read=1 WHERE alert_id=?", (aid,))
        db.commit()
        return jsonify(reviewed=aid)

    # ---------------------------------------------------------------- model re-training, export, delete
    @app.post("/api/retrain")
    @login_required
    def api_retrain():
        db, uid = get_db(), session["user_id"]
        corr = [(r["description"], r["category"]) for r in db.execute("SELECT description, category FROM corrections WHERE user_id=?", (uid,))]
        if not corr:
            return jsonify(error="No corrections yet. Correct a few predicted categories first."), 400
        return jsonify(retrained_with=registry.retrain_user(uid, corr))

    @app.get("/api/export")
    @login_required
    def api_export():
        db, uid = get_db(), session["user_id"]
        u = user_row(db)
        return jsonify(user={"name": u["name"], "email": u["email"], "monthly_income": u["monthly_income"]},
                       transactions=[tx_json(r) for r in db.execute(TX_SQL + " WHERE t.user_id=? ORDER BY t.txn_date", (uid,))],
                       budgets=[dict(r) for r in db.execute("SELECT month, category_id, limit_amount FROM budgets WHERE user_id=?", (uid,))],
                       alerts=[dict(r) for r in db.execute("SELECT level, message, created_at FROM alerts WHERE user_id=?", (uid,))])

    @app.delete("/api/account")
    @login_required
    def api_delete_account():
        d = request.get_json(silent=True) or {}
        db, uid = get_db(), session["user_id"]
        if not check_password_hash(user_row(db)["password_hash"], str(d.get("password", ""))):
            return jsonify(error="Password is incorrect"), 403
        db.execute("DELETE FROM users WHERE user_id=?", (uid,))
        db.commit()
        registry.drop_user(uid)
        session.clear()
        return jsonify(deleted=True)

    @app.get("/api/predict")
    @login_required
    def api_predict():
        """Live category preview for the Add Expense form."""
        text = request.args.get("description", "")
        cat, conf = registry.categoriser_for(session["user_id"]).predict(text)
        return jsonify(category=cat, confidence=round(conf, 2))

    # ---------------------------------------------------------------- errors
    @app.errorhandler(404)
    def _404(_e):
        return (jsonify(error="Not found"), 404) if request.path.startswith("/api/") else ("Page not found", 404)

    @app.errorhandler(413)
    def _413(_e):
        return jsonify(error="File too large (max 2 MB)"), 413

    @app.errorhandler(500)
    def _500(_e):
        return jsonify(error="Internal server error"), 500

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
