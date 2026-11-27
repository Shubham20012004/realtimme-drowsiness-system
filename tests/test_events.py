import unittest
from monitor.events import match_events

class EventTests(unittest.TestCase):
    def test_one_to_one(self):
        r=match_events([(1,2),(4,5)],[1.2,1.4,5],3600)
        self.assertEqual(r["event_recall"],.5)
        self.assertEqual(r["false_alarms_per_hour"],2)
        self.assertAlmostEqual(r["median_latency_s"],.2)
