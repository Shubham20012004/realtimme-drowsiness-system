import argparse
import json
import logging
import os
import time
import uuid

import cv2

from .engine import Engine
from .hardware import Alerts
from .storage import Writer,database,migrate
from .vision import Vision


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video")
    src.add_argument("--camera",type=int,default=0)
    p.add_argument("--port",help="Explicit lab Arduino port or auto")
    p.add_argument("--baud",type=int,default=9600)
    p.add_argument("--width",type=int,default=1280)
    p.add_argument("--height",type=int,default=720)
    p.add_argument("--fps",type=float,default=30)
    p.add_argument("--headless",action="store_true")
    p.add_argument("--participant",default="anonymous")
    p.add_argument("--conditions",default="unspecified")
    p.add_argument("--db",default=os.getenv("DROWSINESS_DB","runs/research.sqlite3"))
    p.add_argument("--model",help="Trusted, locally trained model bundle")
    p.add_argument("--quality-model",help="Trusted local development-fitted tracking model")
    args = p.parse_args()
    quality = None
    if args.quality_model:
        import joblib
        from .quality import acceptable
        quality = joblib.load(args.quality_model)
    logging.basicConfig(level=logging.INFO)
    model = None
    if args.model:
        from .training import load_model
        model = load_model(args.model)
    migrate(args.db)
    session = uuid.uuid4().hex
    vision = Vision()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        for prop,value in ((cv2.CAP_PROP_FRAME_WIDTH,args.width),(cv2.CAP_PROP_FRAME_HEIGHT,args.height),(cv2.CAP_PROP_FPS,args.fps)):
            cap.set(prop,value)
    if not cap.isOpened():
        cap.release()
        vision.close()
        raise SystemExit("Could not open input")
    with database(args.db) as conn:
        conn.execute("INSERT INTO sessions(id,participant,conditions) VALUES(?,?,?)",(session,args.participant,args.conditions))
    output = Writer(args.db,session)
    alerts = Alerts(args.port,args.baud)
    detector = Engine()
    start = time.monotonic()
    count = 0
    try:
        while True:
            ok,frame = cap.read()
            now = time.monotonic()
            if not ok and args.video:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if args.video else now-start
            if ok:
                obs,reason = vision.observe(frame)
            else:
                obs,reason = None,"CAMERA_ERROR"
            if quality is not None and obs is not None and not acceptable(quality,obs):
                obs,reason = None,"TRACKING_DEGRADED"
            row = detector.update(t,obs,reason)
            if model and row["status"]=="RUNNING":
                from .training import infer
                row["prediction"] = infer(model,row)
            count += int(ok)
            row.update(processing_ms=1000*(time.monotonic()-now),actual_fps=count/(time.monotonic()-start),
                       serial_status=alerts.status,dropped_records=output.dropped,recorded_at=time.time())
            alerts.level = {"ALERT":0,"DROWSY":1,"CRITICAL":2}.get(row["state"],2)
            output.put(row)
            if not ok:
                cap.release()
                time.sleep(1)
                cap.open(args.camera)
                continue
            if not args.headless:
                cv2.putText(frame,row["state"] or row["status"],(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(70,230,255),2)
                cv2.imshow("Stationary research monitor",frame)
                key = cv2.waitKey(1)&255
                if key in (27,ord("q")):
                    break
                if key==ord("r"):
                    detector = Engine()
    finally:
        cap.release()
        vision.close()
        cv2.destroyAllWindows()
        alerts.close()
        output.close()
        print(json.dumps(dict(session=session,frames=count,dropped_records=output.dropped)))


if __name__=="__main__":
    main()
