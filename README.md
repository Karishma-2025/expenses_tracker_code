# AI Based Expense Tracker and Budget Prediction

Mini-project code that goes with the project report. A Flask web application that

* **auto-categorises** every expense from its description (TF-IDF character n-grams + Logistic Regression),
* **forecasts** the next 1-6 months of spending per category (lag / rolling-mean / seasonal features + Linear Regression),
* **recommends a budget** from the forecast, income and savings goal (50/30/20 idea: needs are protected, wants are trimmed),
* **detects unusual transactions** (Isolation Forest on amount + hour of day),
* shows **CAUTION (75%) / WARNING (90%) / EXCEEDED (100%)** budget alerts on a dashboard.

> All models are trained on **synthetic data** (real bank data is private). Results show the behaviour of the pipeline, not real-world accuracy.

## 1. Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows        (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python scripts/seed_demo.py      # optional: demo account with 30 months of data
python app.py                    # open http://127.0.0.1:5000
```

Demo login: **demo@example.com / demo1234** (or register a new account).
The first start trains the default models automatically (a few seconds) and stores them in `models/`;
you can also run `python scripts/train_models.py`.

Try the CSV import with `sample_data/sample_statement.csv` (valid) and `sample_data/sample_statement_with_errors.csv`
(shows row-wise error reporting). Dashboard charts use Chart.js from a CDN, so internet is needed for the charts only.

## 2. Project structure

```
app.py                 Flask app factory, pages and REST API
config.py              settings (DB path, model folder, thresholds)
schema.sql             database schema (6 tables of the ER diagram + corrections table)
ml/core.py             Categoriser, forecast_next, recommend_budget, budget_status, AnomalyDetector (report listings 3.2-3.5)
ml/data_gen.py         synthetic data generators (transactions, monthly series, anomalies, demo history)
ml/registry.py         train / save / load models, per-user re-training from corrections
scripts/train_models.py   train default models
scripts/experiments.py    re-run all report experiments  [--charts DIR]
scripts/seed_demo.py      create demo user + saved budget
templates/, static/    dashboard (HTML, CSS, JavaScript using fetch API + Chart.js)
tests/                 pytest: unit tests TC01-TC15, integration tests IT01-IT10 and extra checks
sample_data/           example CSV statements
```

## 3. Features and where they are implemented

| Report feature | Implementation |
|---|---|
| FR1 register / login / logout, income | `/api/register`, `/api/login`, `/api/logout`, `/api/me` (salted password hash, session cookie) |
| FR2 add / edit / delete expense with validation | `POST/PUT/DELETE /api/expenses` |
| FR3 auto-categorisation with confidence | `ml/core.py -> Categoriser`, `GET /api/predict` (live preview) |
| FR4 manual override + re-training | `PUT /api/expenses/<id>` stores a correction, `POST /api/retrain` builds a personal model |
| FR5 CSV import with error report | `POST /api/upload` |
| FR6 reports | `GET /api/summary?month=YYYY-MM`, dashboard |
| FR7 forecast per category | `GET /api/forecast?months=n` (needs >= 6 complete months, otherwise a clear message) |
| FR8 budget recommendation | `POST /api/budget/recommend` (`save: true` stores it), `GET/PUT /api/budget` |
| FR9 alerts 75 / 90 / 100 % | created automatically on every transaction or budget change, `GET /api/alerts` |
| FR10 anomaly detection | Isolation Forest; personal model after 50 transactions, global model before that |
| Privacy | data isolation (HTTP 403 for other users' records), `GET /api/export`, `DELETE /api/account` |

### REST API

| Method | Endpoint | Function |
|---|---|---|
| POST | /api/register, /api/login, /api/logout | account handling |
| GET / PUT | /api/me | profile and monthly income |
| POST / GET | /api/expenses | add / list (filters: month, category) |
| PUT / DELETE | /api/expenses/<id> | correct / delete a transaction |
| POST | /api/upload | CSV import (columns: date, description, amount; optional time, mode, category) |
| GET | /api/summary | category-wise spending, budgets and status |
| GET | /api/forecast | forecast for the next n months |
| POST | /api/budget/recommend | recommended category budget |
| GET / PUT | /api/budget | saved budgets |
| GET / PUT | /api/alerts, /api/alerts/<id>/review | alerts |
| GET | /api/predict | category preview for a description |
| POST | /api/retrain | personal categoriser from corrections |
| GET / DELETE | /api/export, /api/account | download / delete all data |

## 4. Tests

```bash
python -m pytest -q
```

42 tests: the 15 unit tests (TC01-TC15) of the report, the 10 integration scenarios (IT01-IT10) and extra checks
(password hashing, CSV errors, anomaly alert flow, export / delete account ...).

## 5. Reproduce the report results

```bash
python scripts/experiments.py --charts charts
```

Seed 42: Logistic Regression accuracy 100.00%, Naive Bayes 99.79%; forecasting MAE Linear Regression 883 (best),
Gradient Boosting 1033, Moving Average 1081, Random Forest 1150; Isolation Forest F1 0.87; 6-month forecast Rs. 36,292 ... 36,967.

## 6. Notes

* Set the `SECRET_KEY` environment variable and use HTTPS plus a production server (e.g. gunicorn) for real deployment.
  The session cookie is HttpOnly and SameSite=Lax; add CSRF tokens if the API is exposed cross-site.
* SQLite is used for simplicity; `schema.sql` is plain SQL and ports to MySQL / PostgreSQL with minor syntax changes.
* The forecast uses complete past months only; the current (incomplete) month is the first predicted month.
* To train on real behaviour, replace the synthetic data in `ml/data_gen.py` with your own consented data.
