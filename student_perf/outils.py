# utils.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]
CAT = ["Extracurricular Activities"]
TARGET = "Performance Index"

def load_raw(path="data/student_perf.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="if_binary"), CAT),
    ])