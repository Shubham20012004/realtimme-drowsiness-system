"""Extract authorised local video manifests, retaining native annotation granularity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd

from .engine import Engine,FEATURES,SCHEMA
from .vision import Vision


def fingerprint(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def intervals(path):
    data=pd.read_csv(path,dtype={"label":str}).sort_values("start")
    if data.empty or data[["start","end","label"]].isna().any().any() or (data.start<0).any() or (data.end<=data.start).any():
        raise ValueError(f"Invalid annotations in {path}")
    if (data.start.iloc[1:].to_numpy()<data.end.iloc[:-1].to_numpy()).any():
        raise ValueError("Overlapping annotations need adjudication")
    return list(data.itertuples(index=False))


def extract(manifest,output):
    root=Path(manifest).resolve().parent
    records=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    if records.session.duplicated().any() or records.dataset.nunique()!=1:
        raise ValueError("Use one dataset per manifest and one video per session")
    dest=Path(output)
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/"features.csv").open("x",newline="",encoding="utf8") as f:
        cols=["participant","session","source_sha256","split","schema","t","label","label_source","status"]+FEATURES
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        writer.writeheader()
        for r in records.itertuples(index=False):
            path=root/r.video
            identity=dict(participant=r.participant,session=r.session,split=r.split,source_sha256=fingerprint(path),schema=SCHEMA)
            if not r.participant or not r.session:
                raise ValueError("Missing subject or session")
            if r.dataset=="uta-rldd":
                if r.label not in {"0","5","10"}:
                    raise ValueError("Expected UTA video label 0, 5 or 10")
                ann=[]
            elif r.dataset in {"nthu","yawdd","lab"}:
                ann=intervals(root/r.annotation)
            else:
                raise ValueError("Unsupported dataset")
            cap=cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open {path}")
            vision=Vision()
            engine=Engine()
            active=0
            valid=[]
            count=0
            try:
                while True:
                    ok,img=cap.read()
                    if not ok:
                        break
                    t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    obs,reason=vision.observe(img)
                    row=engine.update(t,obs,reason)
                    count+=1
                    if r.dataset=="uta-rldd":
                        if row["status"]=="RUNNING":
                            valid.append({k:row.get(k) for k in FEATURES})
                        continue
                    while active<len(ann) and t>=ann[active].end:
                        active+=1
                    if active<len(ann) and ann[active].start<=t and row["status"]=="RUNNING":
                        writer.writerow((row | identity) | {"label":ann[active].label,"label_source":"independent_interval"})
                if not count:
                    raise ValueError(f"Empty video {path}")
                if r.dataset=="uta-rldd":
                    summary=pd.DataFrame(valid,columns=FEATURES).median().to_dict()
                    writer.writerow(identity | summary | dict(t=0,label=r.label,label_source="video_self_report",status="VIDEO_SUMMARY"))
                with (dest/"inventory.jsonl").open("a",encoding="utf8") as index:
                    index.write(json.dumps(identity | dict(frames=count,dataset=r.dataset,label_scope="video" if r.dataset=="uta-rldd" else "interval"))+"\n")
            finally:
                vision.close()
                cap.release()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("output")
    args=p.parse_args()
    extract(args.manifest,args.output)
