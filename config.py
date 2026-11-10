"""Application configuration."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-in-production")
    DATABASE = os.environ.get("EXPENSE_DB", os.path.join(BASE_DIR, "expense.db"))
    MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
    MIN_FORECAST_MONTHS = 6          # months of history needed for a forecast
    CATEGORY_CONFIDENCE = 0.25       # below this -> "Uncategorised"
    MAX_UPLOAD_BYTES = 2 * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
