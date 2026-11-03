# outils.py
from pathlib import Path
import pandas as pd
import json
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 13
DATA_PATH = Path("student-por.csv")
SPLIT_DIR = Path("split")
PARAMS_DIR = Path("params")
TARGET = "G3"

NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
               "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
               "nursery", "higher", "internet", "romantic"]

# Les 3 scénarios de variables à comparer
SCENARIOS = {
    "A_G1_G2": NUMERIC + CATEGORICAL + ["G1", "G2"],
    "B_G1": NUMERIC + CATEGORICAL + ["G1"],
    "C_sans_notes": NUMERIC + CATEGORICAL,
}


def load_data(path=DATA_PATH):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def save_split(X_train, X_test, y_train, y_test, folder=SPLIT_DIR):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(folder / "X_train.csv", index=False)
    X_test.to_csv(folder / "X_test.csv", index=False)
    y_train.to_csv(folder / "y_train.csv", index=False)
    y_test.to_csv(folder / "y_test.csv", index=False)


def load_split(folder=SPLIT_DIR):
    folder = Path(folder)
    X_train = pd.read_csv(folder / "X_train.csv")
    X_test = pd.read_csv(folder / "X_test.csv")
    y_train = pd.read_csv(folder / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(folder / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test

def save_params(nom, params, folder=PARAMS_DIR):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{nom}.csv").write_text(json.dumps(params, indent=2, ensure_ascii=False, default=lambda o: o.item()))

def load_params(nom, folder=PARAMS_DIR):
    folder = Path(folder)
    return json.loads((folder / f"{nom}.csv").read_text(encoding="utf-8-sig"))


def build_preprocessor(features):
    num = [c for c in NUMERIC + ["G1", "G2"] if c in features]
    cat = [c for c in CATEGORICAL if c in features]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])  # les colonnes non listées sont ignorées (remainder="drop")


def make_pipeline(model, features):
    return Pipeline([("prep", build_preprocessor(features)), ("model", model)])


def evaluate_cv(model, X, y, features, n_splits=5):
    """Validation croisée sur le train : retourne RMSE, MAE, R² moyens."""
    cv = KFold(n_splits, shuffle=True, random_state=SEED)
    s = cross_validate(
        make_pipeline(model, features), X, y, cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error",
                 "mae": "neg_mean_absolute_error", "r2": "r2"})
    return {"rmse": -s["test_rmse"].mean(), "rmse_std": s["test_rmse"].std(),
            "mae": -s["test_mae"].mean(), "r2": s["test_r2"].mean()}


def evaluate_test(pipe, X_test, y_test):
    """À n'utiliser qu'une fois, dans 4-evaluation."""
    pred = pipe.predict(X_test)
    return {"rmse": mean_squared_error(y_test, pred) ** 0.5,
            "mae": mean_absolute_error(y_test, pred),
            "r2": r2_score(y_test, pred)}