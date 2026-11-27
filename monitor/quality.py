"""Optional development-only tracking geometry model. Behaviour features are excluded."""
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

COLUMNS=["brightness","blur","reprojection_error"]


def fit(path,out):
    df=pd.read_csv(path)
    if not df.split.eq("development").all() or not df.quality_label.eq("valid").all():
        raise ValueError("Expected development-only independently accepted tracking examples")
    if df.participant.nunique()<2 or len(df)<100:
        raise ValueError("Insufficient calibration examples")
    model=IsolationForest(n_estimators=128,contamination="auto",random_state=23)
    model.fit(df[COLUMNS].dropna())
    joblib.dump({"columns":COLUMNS,"model":model},out)


def acceptable(bundle,obs):
    if bundle["columns"]!=COLUMNS:
        raise ValueError("Quality model schema mismatch")
    sample=pd.DataFrame([{k:obs[k] for k in COLUMNS}])
    return bool(bundle["model"].predict(sample)[0]==1)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("output")
    a=p.parse_args()
    fit(a.csv,a.output)
