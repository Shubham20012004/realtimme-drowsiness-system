import logging
import threading
import time

import serial
from serial.tools import list_ports


class Alerts:
    """Single worker owns the serial port. Only the most recent state is relevant."""
    def __init__(self, port, baud):
        self.port,self.baud = port,baud
        self.level = None
        self.status = "DISABLED" if not port else "DISCONNECTED"
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run)
        if port:
            self.thread.start()

    def run(self):
        while not self.done.is_set():
            port = self.port
            if port == "auto":
                candidates = [p.device for p in list_ports.comports() if "arduino" in (p.description or "").lower()]
                port = candidates[0] if len(candidates)==1 else None
            if port:
                try:
                    with serial.Serial(port,self.baud,timeout=0.1,write_timeout=0.2) as device:
                        self.status = "CONNECTED"
                        self.done.wait(2)
                        sent = None
                        while not self.done.is_set():
                            level = self.level
                            if level is not None and level != sent:
                                before = time.monotonic()
                                device.write(str(level).encode("ascii"))
                                logging.info("serial_write code=%s elapsed_s=%f",level,time.monotonic()-before)
                                sent = level
                            self.done.wait(0.03)
                        device.write(b"3")
                except (OSError,serial.SerialException) as exc:
                    logging.warning("Serial %s disconnected: %s",port,exc)
            self.status = "DISCONNECTED"
            self.done.wait(1)

    def close(self):
        self.done.set()
        if self.thread.is_alive():
            self.thread.join()
