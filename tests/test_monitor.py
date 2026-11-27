import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from monitor.engine import Engine,ear
from monitor.storage import Writer,database,migrate


def obs(left=.3):
    return dict(left=left,right=left,mar=.04,pitch=0,yaw=0,roll=0)


class EngineTests(unittest.TestCase):
    def calibrated(self):
        e = Engine(calibration_s=1,window=2)
        for i in range(21):
            row = e.update(i/20,obs())
        self.assertEqual(row["status"],"RUNNING")
        return e

    def test_geometry(self):
        self.assertEqual(ear([(0,0),(1,1),(3,1),(4,0),(3,-1),(1,-1)]),.5)
        with self.assertRaises(ValueError):
            ear([(0,0)]*6)

    def test_critical_and_loss(self):
        e = self.calibrated()
        for i in range(21,61):
            row = e.update(i/20,obs(.08))
        self.assertEqual(row["state"],"CRITICAL")
        self.assertGreater(row["perclos"],.9)
        self.assertIsNone(e.update(3.1,None)["state"])
        self.assertEqual(e.update(3.2,obs(.08))["closure_s"],0)
        self.assertIsNone(e.update(7,obs())["perclos"])

    def test_recovery_requires_persistence(self):
        e=self.calibrated()
        for i in range(21,57):
            row=e.update(i/20,obs(.08))
        self.assertEqual(row["state"],"CRITICAL")
        self.assertEqual(e.update(2.9,obs())["state"],"CRITICAL")
        self.assertEqual(e.update(3.16,obs())["state"],"ALERT")

    def test_duplicate_and_invalid_time(self):
        e = self.calibrated()
        for t in (1,.9,float("nan")):
            with self.assertRaises(ValueError):
                e.update(t,obs())

    def test_reject_closed_calibration(self):
        e = Engine(calibration_s=1)
        for i in range(21):
            row = e.update(i/20,obs(.02))
        self.assertEqual(row["status"],"CALIBRATION_RETRY")
        self.assertIsNone(e.baseline)

    def test_blink(self):
        e = self.calibrated()
        for i in range(21,66):
            row = e.update(i/20,obs(.1 if 40<=i<43 else .3))
        self.assertAlmostEqual(row["mean_blink_s"],.15)
        self.assertAlmostEqual(row["blink_per_min"],30)


class StorageTests(unittest.TestCase):
    def test_flush_and_migrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"run.db"
            migrate(path)
            migrate(path)
            with database(path) as db:
                db.execute("INSERT INTO sessions(id) VALUES('demo')")
            w = Writer(path,"demo")
            for i in range(37):
                w.put(dict(t=i,status="TRACKING_LOST"))
            w.close()
            with database(path) as db:
                self.assertEqual(db.execute("select count(*) from frames").fetchone()[0],37)
                self.assertEqual(db.execute("pragma integrity_check").fetchone()[0],"ok")

    def test_protect_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"archive.db"
            with database(path) as db:
                db.execute("CREATE TABLE telemetry(x)")
            with self.assertRaises(ValueError):
                migrate(path)
