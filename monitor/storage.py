import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue

DDL = """
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, participant TEXT, conditions TEXT, started TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS frames(session TEXT REFERENCES sessions(id), t REAL, payload TEXT NOT NULL, PRIMARY KEY(session,t));
CREATE TABLE IF NOT EXISTS annotations(id INTEGER PRIMARY KEY,session TEXT REFERENCES sessions(id),start REAL,end REAL,label TEXT,observer TEXT,replaces INTEGER REFERENCES annotations(id),CHECK(end>start));
"""


@contextmanager
def database(path=None):
    path = Path(path or os.getenv("DROWSINESS_DB","runs/research.sqlite3"))
    if path.name == "drowsiness_telemetry.db":
        raise ValueError("Select a separate research database")
    path.parent.mkdir(parents=True,exist_ok=True)
    conn = sqlite3.connect(path,timeout=5)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            yield conn
    finally:
        conn.close()


def migrate(path=None):
    with database(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise ValueError("Newer database version")
        if version == 0 and conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]:
            raise ValueError("Existing unversioned database. Select a new research database.")
        for statement in DDL.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.execute("PRAGMA user_version=1")


class Writer:
    def __init__(self, path, session):
        self.path,self.session = path,session
        self.queue = Queue(512)
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def put(self, row):
        if self.error:
            raise RuntimeError("Telemetry writer failed") from self.error
        try:
            self.queue.put_nowait((self.session,row["t"],json.dumps(row,allow_nan=False)))
        except Full:
            self.dropped += 1

    def run(self):
        try:
            with database(self.path) as conn:
                while True:
                    first = self.queue.get()
                    if first is None:
                        break
                    batch = [first]
                    stop = False
                    while len(batch)<64:
                        try:
                            item = self.queue.get_nowait()
                        except Empty:
                            break
                        if item is None:
                            stop = True
                            break
                        batch.append(item)
                    with conn:
                        conn.executemany("INSERT INTO frames VALUES(?,?,?)",batch)
                    if stop:
                        break
        except Exception as error:
            self.error = error

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None,timeout=0.1)
                break
            except Full:
                continue
        self.thread.join()
        if self.error:
            raise RuntimeError("Telemetry flush failed") from self.error
