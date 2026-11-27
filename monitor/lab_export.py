"""Convert adjudicated independent intervals to a training table using an explicit split manifest."""
import argparse
import json

import pandas as pd

from .engine import FEATURES
from .storage import database
from .training import validate


def export(manifest,db_path,dest):
    specs=pd.read_csv(manifest,dtype=str,keep_default_na=False)
    records=[]
    with database(db_path) as db:
        for s in specs.itertuples(index=False):
            info=db.execute("SELECT participant,conditions FROM sessions WHERE id=?",(s.session,)).fetchone()
            if not info or info[0]=="anonymous":
                raise ValueError("A stable pseudonymous participant code is required")
            notes=db.execute("""SELECT a.start,a.end,a.label FROM annotations a
                WHERE a.session=? AND NOT EXISTS(SELECT 1 FROM annotations b WHERE b.replaces=a.id)
                ORDER BY a.start""",(s.session,)).fetchall()
            if any(a[1]>b[0] for a,b in zip(notes,notes[1:])):
                raise ValueError("Adjudicate overlapping independent annotations before export")
            i=0
            for t,payload in db.execute("SELECT t,payload FROM frames WHERE session=? ORDER BY t",(s.session,)):
                while i<len(notes) and t>=notes[i][1]:
                    i+=1
                if i==len(notes):
                    break
                row=json.loads(payload)
                if row["status"]=="RUNNING" and t>=notes[i][0]:
                    records.append({k:row.get(k) for k in FEATURES}|dict(t=t,schema=row["schema"],participant=info[0],
                        conditions=info[1],session=s.session,split=s.split,source_sha256=s.source_sha256,
                        label=notes[i][2],label_source="independent_interval"))
    df=pd.DataFrame(records)
    validate(df)
    df.to_csv(dest,index=False,mode="x")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("manifest",help="CSV columns session,split,source_sha256")
    p.add_argument("database")
    p.add_argument("output")
    a=p.parse_args()
    export(a.manifest,a.database,a.output)
