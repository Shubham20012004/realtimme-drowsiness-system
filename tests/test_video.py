import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from monitor.storage import database


class VideoTests(unittest.TestCase):
    def test_unobservable_video_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root=Path(tmp).resolve()
            video=root/"no-face.avi"
            w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),10,(128,128))
            self.assertTrue(w.isOpened())
            for i in range(20):
                w.write(np.full((128,128,3),i,dtype=np.uint8))
            w.release()
            db=root/"session.sqlite3"
            result=subprocess.run([sys.executable,"-m","monitor.run","--video",str(video),
                                   "--headless","--db",str(db)],capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr[-1500:])
            self.assertEqual(json.loads(result.stdout.splitlines()[-1])["dropped_records"],0)
            with database(db) as conn:
                rows=[json.loads(r[0]) for r in conn.execute("SELECT payload FROM frames ORDER BY t")]
            self.assertEqual(len(rows),20)
            self.assertTrue(all(r["status"]=="TRACKING_LOST" and r["state"] is None for r in rows))
            self.assertAlmostEqual(rows[-1]["t"],1.9)
