import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from monitor.engine import FEATURES,SCHEMA
from monitor.training import infer,load_model,train,validate


def fixture():
    rows=[]
    for i in range(8):
        for j in range(4):
            row = dict.fromkeys(FEATURES,0.1+j%2)
            row.update(participant=str(i),session=str(i),t=j,source_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                       label=str(j%2),schema=SCHEMA,split="test" if i>=6 else "development",label_source="independent_interval")
            rows.append(row)
    return pd.DataFrame(rows)


class TrainingTests(unittest.TestCase):
    def test_leakage(self):
        df = fixture()
        df.loc[0,"participant"]="6"
        with self.assertRaises(ValueError):
            validate(df)

    def test_predictions_are_ineligible_labels(self):
        df = fixture()
        df.loc[0,"label_source"]="prediction"
        with self.assertRaises(ValueError):
            validate(df)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            df=fixture()
            csv=Path(tmp)/"fixture.csv"
            out=Path(tmp)/"bundle"
            df.to_csv(csv,index=False)
            train(csv,out)
            b=load_model(out/"model.joblib")
            row=dict(df.iloc[0],status="RUNNING")
            self.assertIn(infer(b,row),["0","1"])
            row["status"]="TRACKING_LOST"
            self.assertIsNone(infer(b,row))
