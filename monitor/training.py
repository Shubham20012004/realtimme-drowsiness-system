"""Independent interval-labelled feature tables only. See research/PROTOCOL.md."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .engine import FEATURES,SCHEMA

SETS = {"ear":["left","right"],"ear_mar":["left","right","mar"],
        "perclos":["left","right","mar","perclos"],"full":FEATURES}


def validate(df):
    required = set(FEATURES+["participant","session","source_sha256","label","split","label_source","schema","t"])
    missing = required-set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}")
    identifiers = ["participant","session","source_sha256","label","split"]
    if df[identifiers].isna().any().any() or not df.schema.eq(SCHEMA).all():
        raise ValueError("Missing identities or incompatible schema")
    if not df.label_source.eq("independent_interval").all():
        raise ValueError("Independent interval annotations required")
    if set(df.split)!={"development","test"} or not df.source_sha256.str.fullmatch("[a-f0-9]{64}").all():
        raise ValueError("Invalid split or video fingerprint")
    for key in identifiers[:3]:
        if df.groupby(key).split.nunique().max()>1:
            raise ValueError(f"Partition leakage: {key}")
    for key in ("session","source_sha256"):
        if df.groupby(key).participant.nunique().max()>1:
            raise ValueError(f"Inconsistent participant for {key}")
    if df.duplicated(["session","t"]).any():
        raise ValueError("Duplicate session timestamp")


def load_model(path):
    bundle = joblib.load(path)
    if bundle["schema"]!=SCHEMA or bundle["sklearn"]!=sklearn.__version__ or bundle["features"] not in SETS.values():
        raise ValueError("Incompatible model")
    return bundle


def infer(bundle,row):
    cols = bundle["features"]
    if row.get("schema")!=SCHEMA or row.get("status")!="RUNNING" or any(pd.isna(row.get(k)) for k in cols):
        return None
    return str(bundle["estimator"].predict(pd.DataFrame([{k:row[k] for k in cols}]))[0])


def train(csv, destination):
    df = pd.read_csv(csv,dtype={k:str for k in ("participant","session","source_sha256","label")})
    validate(df)
    dev = df[df.split=="development"]
    labels = sorted(dev.label.unique())
    if len(labels)<2 or dev.participant.nunique()<3:
        raise ValueError("At least two classes and three development subjects required")
    cv = list(StratifiedGroupKFold(3,shuffle=True,random_state=23).split(dev,dev.label,dev.participant))
    for tr,va in cv:
        if set(dev.iloc[tr].label)!=set(labels) or set(dev.iloc[va].label)!=set(labels):
            raise ValueError("Every validation and training fold must cover every class")
    candidates = {"logistic":LogisticRegression(max_iter=2000,class_weight="balanced"),
                  "tree":DecisionTreeClassifier(max_depth=5,class_weight="balanced",random_state=23),
                  "forest":RandomForestClassifier(n_estimators=100,min_samples_leaf=3,class_weight="balanced",random_state=23),
                  "svm":SVC(class_weight="balanced")}
    scores = []
    best = None
    for name,estimator in candidates.items():
        for features,cols in SETS.items():
            pipe = make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),estimator)
            values = cross_val_score(pipe,dev[cols],dev.label,cv=cv,scoring="f1_macro",error_score="raise")
            scores.append(dict(model=name,features=features,fold_macro_f1=values.tolist()))
            if best is None or values.mean()>best[0]:
                best = (values.mean(),name,cols,pipe)
    _,name,cols,pipe = best
    pipe.fit(dev[cols],dev.label)
    test = df[df.split=="test"]
    if set(test.label)!=set(labels):
        raise ValueError("Test label set differs from development label set")
    eligible = test[cols].notna().all(axis=1)
    if not eligible.any():
        raise ValueError("No eligible held-out observations")
    test_valid = test.loc[eligible]
    pred = pipe.predict(test_valid[cols])
    bundle = dict(schema=SCHEMA,features=cols,model=name,estimator=pipe,sklearn=sklearn.__version__)
    report = dict(model=name,features=cols,schema=SCHEMA,labels=labels,seed=23,
                  dataset_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest(),development_cv=scores,
                  participants={s:sorted(g.participant.unique()) for s,g in df.groupby("split")},
                  eligible_fraction=float(eligible.mean()),n_eligible=int(eligible.sum()),
                  balanced_accuracy=balanced_accuracy_score(test_valid.label,pred),
                  classification=classification_report(test_valid.label,pred,labels=labels,output_dict=True,zero_division=0),
                  confusion=confusion_matrix(test_valid.label,pred,labels=labels).tolist())
    dest = Path(destination)
    dest.mkdir(parents=True,exist_ok=False)
    joblib.dump(bundle,dest/"model.joblib")
    (dest/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("output")
    args = p.parse_args()
    train(args.csv,args.output)
