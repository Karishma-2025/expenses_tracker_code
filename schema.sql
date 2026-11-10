-- AI Based Expense Tracker and Budget Prediction : database schema (SQLite)
CREATE TABLE IF NOT EXISTS users (
    user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    email          TEXT UNIQUE NOT NULL,
    password_hash  TEXT NOT NULL,
    monthly_income REAL DEFAULT 0,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    type        TEXT CHECK (type IN ('Need','Want'))
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category_id  INTEGER REFERENCES categories(category_id),   -- NULL = Uncategorised
    amount       REAL NOT NULL CHECK (amount > 0),
    description  TEXT NOT NULL,
    txn_date     DATE NOT NULL,
    txn_hour     INTEGER DEFAULT 12,
    payment_mode TEXT,
    confidence   REAL,
    is_anomaly   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_txn_user_date ON transactions(user_id, txn_date);
CREATE TABLE IF NOT EXISTS budgets (
    budget_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category_id  INTEGER NOT NULL REFERENCES categories(category_id),
    month        TEXT NOT NULL,                               -- YYYY-MM
    limit_amount REAL NOT NULL,
    UNIQUE (user_id, category_id, month)
);
CREATE TABLE IF NOT EXISTS predictions (
    pred_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category_id   INTEGER NOT NULL REFERENCES categories(category_id),
    target_month  TEXT NOT NULL,
    predicted_amt REAL NOT NULL,
    model_name    TEXT,
    UNIQUE (user_id, category_id, target_month)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    message    TEXT NOT NULL,
    level      TEXT NOT NULL,                                 -- CAUTION / WARNING / EXCEEDED / ANOMALY
    ref_key    TEXT,
    is_read    INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, ref_key)
);
-- user corrections of the predicted category, used to re-train the personal categoriser
CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    category    TEXT NOT NULL,
    UNIQUE (user_id, description)
);
