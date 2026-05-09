import pickle
from pathlib import Path


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    path = Path(path)

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model