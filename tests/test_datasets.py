import tempfile
import unittest
from pathlib import Path
from monitor.datasets import intervals
from monitor.quality import COLUMNS

class DatasetTests(unittest.TestCase):
    def test_interval_scope(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"annotations.csv"
            p.write_text("start,end,label\n0,.5,blink\n.5,2,open\n")
            self.assertEqual([a.label for a in intervals(p)],["blink","open"])
            p.write_text("start,end,label\n0,1,blink\n.5,2,open\n")
            with self.assertRaises(ValueError):
                intervals(p)

    def test_quality_inputs_exclude_eye_mouth_and_pose_state(self):
        self.assertEqual(COLUMNS,["brightness","blur","reprojection_error"])
