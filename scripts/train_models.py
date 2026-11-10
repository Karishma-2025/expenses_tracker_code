"""Train and save the default models.   Usage:  python scripts/train_models.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from ml.registry import train_default

if __name__ == "__main__":
    cat, det = train_default(Config.MODEL_DIR, threshold=Config.CATEGORY_CONFIDENCE)
    print(f"Models saved in {Config.MODEL_DIR}")
    for text in ["swiggy dinner upi", "uber ride", "electricity bill", "netflix subscription", "apollo pharmacy"]:
        print(f"  {text!r:28} -> {cat.predict(text)}")
    print("  anomaly(150000, 3 AM) ->", det.is_anomaly(150000, 3), "| anomaly(250, 2 PM) ->", det.is_anomaly(250, 14))
