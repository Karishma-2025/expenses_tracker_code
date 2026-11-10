import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app
from ml.registry import train_default


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("models"))
    train_default(d)
    return d


@pytest.fixture()
def app(tmp_path, model_dir):
    return create_app({"TESTING": True, "DATABASE": str(tmp_path / "test.db"), "MODEL_DIR": model_dir, "SECRET_KEY": "test"})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def user(client):
    r = client.post("/api/register", json={"name": "Asha", "email": "asha@example.com", "password": "secret123", "monthly_income": 40000})
    assert r.status_code == 201
    return client
