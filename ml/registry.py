"""Training, saving, loading and re-training of the ML models."""
import os
import threading

import joblib
import numpy as np

from .core import Categoriser, AnomalyDetector
from .data_gen import generate_transactions, generate_anomaly_data

CATEGORISER_FILE = "categoriser.joblib"
ANOMALY_FILE = "anomaly.joblib"


def train_default(model_dir, seed=42, threshold=0.25):
    """Train the global categoriser and anomaly detector on synthetic data and save them."""
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tx = generate_transactions(rng)
    cat = Categoriser(threshold=threshold).fit(tx.description, tx.category)
    amt, hr, _ = generate_anomaly_data(np.random.default_rng(seed + 1))
    det = AnomalyDetector().fit(amt, hr)
    joblib.dump(cat, os.path.join(model_dir, CATEGORISER_FILE))
    joblib.dump(det, os.path.join(model_dir, ANOMALY_FILE))
    return cat, det


class ModelRegistry:
    """Loads models lazily; keeps one personal categoriser per user once they have corrected categories."""

    def __init__(self, model_dir, threshold=0.25):
        self.model_dir = model_dir
        self.threshold = threshold
        self._lock = threading.Lock()
        self._global_cat = None
        self._detector = None
        self._personal = {}

    def _ensure(self):
        with self._lock:
            if self._global_cat is None:
                cp = os.path.join(self.model_dir, CATEGORISER_FILE)
                ap = os.path.join(self.model_dir, ANOMALY_FILE)
                if not (os.path.exists(cp) and os.path.exists(ap)):
                    train_default(self.model_dir, threshold=self.threshold)
                self._global_cat = joblib.load(cp)
                self._detector = joblib.load(ap)
                self._global_cat.threshold = self.threshold

    def categoriser_for(self, user_id=None):
        self._ensure()
        if user_id is not None:
            if user_id in self._personal:
                return self._personal[user_id]
            path = os.path.join(self.model_dir, f"user_{user_id}.joblib")
            if os.path.exists(path):
                self._personal[user_id] = joblib.load(path)
                return self._personal[user_id]
        return self._global_cat

    @property
    def detector(self):
        self._ensure()
        return self._detector

    def retrain_user(self, user_id, corrections, repeat=5, seed=42):
        """Base synthetic data + the user's corrected (description, category) pairs (weighted by `repeat`)."""
        self._ensure()
        base = generate_transactions(np.random.default_rng(seed))
        texts = list(base.description) + [d for d, _ in corrections] * repeat
        labels = list(base.category) + [c for _, c in corrections] * repeat
        model = Categoriser(threshold=self.threshold).fit(texts, labels)
        joblib.dump(model, os.path.join(self.model_dir, f"user_{user_id}.joblib"))
        self._personal[user_id] = model
        return len(corrections)

    def drop_user(self, user_id):
        self._personal.pop(user_id, None)
        path = os.path.join(self.model_dir, f"user_{user_id}.joblib")
        if os.path.exists(path):
            os.remove(path)
