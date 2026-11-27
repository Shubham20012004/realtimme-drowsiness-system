import json
import sqlite3
import tempfile
import unittest
import os
from pathlib import Path


class LegacyTests(unittest.TestCase):
    def test_last_partial_batch(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                import ultimatedrowsiness as legacy
                log=legacy.AsyncTelemetryLogger()
                for _ in range(7):
                    log.log_frame({"ear":.3,"state":"test fixture"})
                log.cleanup()
                conn=sqlite3.connect("drowsiness_telemetry.db")
                try:
                    self.assertEqual(conn.execute("SELECT count(*) FROM telemetry").fetchone()[0],7)
                finally:
                    conn.close()
                self.assertEqual(len(next(Path(".").glob("*.jsonl")).read_text().splitlines()),7)
            finally:
                os.chdir(previous)
