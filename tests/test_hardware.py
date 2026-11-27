import threading
import time
import unittest
from unittest.mock import patch
from monitor.hardware import Alerts

class MockSerial:
    def __init__(self):
        self.codes=[]
        self.received=threading.Event()
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass
    def write(self,code):
        self.codes.append(code)
        self.received.set()
        return len(code)

class SerialTests(unittest.TestCase):
    def test_deduplicate_and_shutdown(self):
        device=MockSerial()
        with patch("monitor.hardware.serial.Serial",return_value=device):
            alerts=Alerts("fixture",9600)
            try:
                alerts.level=2
                self.assertTrue(device.received.wait(4))
                time.sleep(.1)
            finally:
                alerts.close()
        self.assertEqual(device.codes,[b"2",b"3"])
