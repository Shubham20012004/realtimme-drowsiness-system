import csv
import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from .storage import database,migrate


@asynccontextmanager
async def lifespan(app):
    migrate()
    yield


app = FastAPI(title="Driver monitoring research",lifespan=lifespan)


class Annotation(BaseModel):
    session: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    label: str = Field(min_length=1,max_length=64)
    observer: str = Field(min_length=1,max_length=64)
    replaces: int | None = None


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).with_name("dashboard.html"))


@app.get("/health")
def health():
    with database() as db:
        return {"database_version":db.execute("PRAGMA user_version").fetchone()[0]}


@app.get("/sessions")
def sessions():
    with database() as db:
        return [dict(zip(("id","participant","conditions","started"),row)) for row in db.execute("SELECT * FROM sessions ORDER BY started DESC,id DESC")]


@app.get("/sessions/{sid}/frames")
def frames(sid:str, limit:int=Query(300,ge=1,le=2000)):
    with database() as db:
        records = db.execute("SELECT payload FROM frames WHERE session=? ORDER BY t DESC LIMIT ?",(sid,limit)).fetchall()
    return [json.loads(row[0]) for row in reversed(records)]


@app.get("/sessions/{sid}/latest")
def latest(sid:str):
    values = frames(sid,1)
    if not values:
        return {"status":"INSUFFICIENT_DATA","state":None}
    row = values[0]
    if time.time()-row["recorded_at"]>3:
        return dict(row,status="STALE",state=None,prediction=None)
    return row


@app.post("/annotations",status_code=201)
def annotate(a:Annotation):
    if a.end<=a.start:
        raise HTTPException(422,"End must follow start")
    with database() as db:
        last = db.execute("SELECT MAX(t) FROM frames WHERE session=?",(a.session,)).fetchone()[0]
        if last is None or a.end>last:
            raise HTTPException(422,"Interval outside recorded session")
        if a.replaces and not db.execute("SELECT 1 FROM annotations WHERE session=? AND id=?",(a.session,a.replaces)).fetchone():
            raise HTTPException(422,"Unknown prior annotation")
        cur = db.execute("INSERT INTO annotations(session,start,end,label,observer,replaces) VALUES(?,?,?,?,?,?)",
                         (a.session,a.start,a.end,a.label,a.observer,a.replaces))
        return {"id":cur.lastrowid}


@app.get("/sessions/{sid}/annotations")
def annotations(sid:str):
    with database() as db:
        return [dict(zip(("id","start","end","label","observer","replaces"),row)) for row in db.execute("SELECT id,start,end,label,observer,replaces FROM annotations WHERE session=? ORDER BY id",(sid,))]


@app.get("/sessions/{sid}/export.csv")
def export(sid:str):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["session","t","payload_json"])
    with database() as db:
        writer.writerows(db.execute("SELECT session,t,payload FROM frames WHERE session=? ORDER BY t",(sid,)))
    return Response(out.getvalue(),media_type="text/csv")

@app.get("/evaluations")
def evaluations():
    import os
    root=Path(os.environ.get("DROWSINESS_RESULTS","runs"))
    results=[]
    for report in sorted(root.glob("*/evaluation.json")):
        try:
            record=json.loads(report.read_text(encoding="utf8"))
            results.append({"run":report.parent.name,"model":record["model"],
                "macro_f1":record["classification"]["macro avg"]["f1-score"],
                "coverage":record["eligible_fraction"],"participants":len(record["participants"]["test"]),
                "confusion":record["confusion"],"labels":record["labels"]})
        except (ValueError,KeyError,OSError) as error:
            results.append({"run":report.parent.name,"error":str(error)})
    return results


@app.get("/sessions/{sid}/stream")
async def stream(sid:str):
    import asyncio
    from fastapi.responses import StreamingResponse
    async def messages():
        previous=None
        while True:
            row=await asyncio.to_thread(latest,sid)
            signature=(row.get("t"),row.get("status"))
            if signature!=previous:
                yield "data: "+json.dumps(row)+"\n\n"
                previous=signature
            else:
                yield ": heartbeat\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(messages(),media_type="text/event-stream")
