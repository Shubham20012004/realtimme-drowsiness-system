import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import uvicorn
from monitor.api import app
from monitor.storage import database,migrate


class ApiTests(unittest.TestCase):
    def test_http_session_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"DROWSINESS_DB":str(Path(tmp)/"test.db")}):
            migrate()
            with database() as db:
                db.execute("INSERT INTO sessions(id) VALUES('s')")
                row=json.dumps(dict(t=3,status="TRACKING_LOST",state=None,recorded_at=0))
                db.execute("INSERT INTO frames VALUES('s',3,?)",(row,))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0))
                port=sock.getsockname()[1]
                server=uvicorn.Server(uvicorn.Config(app,log_level="error"))
                worker=threading.Thread(target=server.run,kwargs={"sockets":[sock]})
                worker.start()
                try:
                    for _ in range(200):
                        if server.started:
                            break
                        time.sleep(.01)
                    url=f"http://127.0.0.1:{port}"
                    def get(path):
                        with urlopen(url+path,timeout=4) as response:
                            return response.read()
                    self.assertEqual(json.loads(get("/health"))["database_version"],1)
                    self.assertIn(b"Driver monitoring lab",get("/"))
                    self.assertEqual(json.loads(get("/sessions/s/latest"))["status"],"STALE")
                    self.assertIn(b"TRACKING_LOST",get("/sessions/s/export.csv"))
                    data=dict(session="s",start=1,end=2,label="closure",observer="a")
                    def post():
                        request=Request(url+"/annotations",data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
                        with urlopen(request,timeout=4) as response:
                            return response.status
                    self.assertEqual(post(),201)
                    self.assertEqual(len(json.loads(get("/sessions/s/annotations"))),1)
                    data["end"]=.5
                    with self.assertRaises(HTTPError) as error:
                        post()
                    self.assertEqual(error.exception.code,422)
                finally:
                    server.should_exit=True
                    worker.join(5)
