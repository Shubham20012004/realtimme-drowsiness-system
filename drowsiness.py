import cv2
import mediapipe as mp
import serial
import time
import numpy as np
from collections import deque
import sys
import threading

# Prevent Windows emoji crash
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def connect_arduino(default_port="COM5"):
    ports_to_try = [default_port]
    if sys.platform.startswith('win'):
        ports_to_try += [f'COM{i}' for i in range(1, 256) if f'COM{i}' != default_port]
    
    for port in ports_to_try:
        try:
            print(f"Checking {port}...")
            s = serial.Serial(port, timeout=0.1, write_timeout=0.1)
            s.close()
            print(f"✓ Arduino connected on: {port}")
            return port
        except (OSError, serial.SerialException):
            pass
    print("⚠ No Arduino detected. Running camera only.")
    return None

class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, backend)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (grabbed, frame) = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

class DrowsinessDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    # === FLAWLESS TIME-BASED THRESHOLDS (in seconds) ===
    DROWSY_TIME = 0.5    # 0.5s = Microsleep (Motor slows, beeps)
    CRITICAL_TIME = 1.5  # 1.5s = Actual sleep (Motor stops, solid alarm)
    CALIB_FRAMES = 90    

    def __init__(self, com_port):
        try:
            self.arduino = serial.Serial(com_port, 9600, timeout=0.1) if com_port else None
            time.sleep(2)
        except:
            self.arduino = None

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        self.ear_baseline = None
        self.calib_data = []
        self.ear_buffer = deque(maxlen=3) # Smoothing buffer for camera noise
        
        self.closure_start_time = None
        self.closure_duration = 0.0
        
        self.state = "CALIBRATING"
        self.current_alert_code = 0
        self.state_color = (0, 255, 0)
        self.last_serial_time = 0

    @staticmethod
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_ear(self, eye, w, h):
        pts = np.array([[lm.x * w, lm.y * h] for lm in eye])
        A = self.distance(pts[1], pts[5])
        B = self.distance(pts[2], pts[4])
        C = self.distance(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0

    def process_frame(self, frame):
        if frame is None: return None
        h, w, _ = frame.shape
        canvas = frame.copy()
        current_time = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        # === NO FACE DETECTED FALLBACK ===
        if not results.multi_face_landmarks:
            self.state = "NO FACE"
            self.current_alert_code = 2
            self.state_color = (0, 0, 255)
            self._send_alert('2')
            self._draw_ui(canvas, h, w, 0)
            return canvas

        landmarks = results.multi_face_landmarks[0].landmark
        
        raw_left = self.calculate_ear([landmarks[i] for i in self.LEFT_EYE], w, h)
        raw_right = self.calculate_ear([landmarks[i] for i in self.RIGHT_EYE], w, h)
        raw_ear = (raw_left + raw_right) / 2.0
        
        # Add to smoothing buffer
        self.ear_buffer.append(raw_ear)
        smoothed_ear = np.mean(self.ear_buffer)

        # === CALIBRATION ===
        if self.ear_baseline is None:
            self.calib_data.append(smoothed_ear)
            progress = len(self.calib_data) / self.CALIB_FRAMES
            self._draw_calibration(canvas, h, w, progress)
            
            if len(self.calib_data) >= self.CALIB_FRAMES:
                self.ear_baseline = np.median(self.calib_data) * 0.82 # 82% threshold
                self.state = "SAFE"
            return canvas

        # === FLAWLESS TIME-BASED LOGIC ===
        if smoothed_ear < self.ear_baseline:
            # Eyes just closed! Start the stopwatch.
            if self.closure_start_time is None:
                self.closure_start_time = current_time
            
            # Calculate how long they've been closed
            self.closure_duration = current_time - self.closure_start_time
        else:
            # Eyes are open! Reset the stopwatch.
            self.closure_start_time = None
            self.closure_duration = 0.0

        # === 3-MODE DECISION ENGINE ===
        if self.closure_duration >= self.CRITICAL_TIME:
            self.state = "CRITICAL (SLEEPING)"
            self.current_alert_code = 2
            self.state_color = (0, 0, 255) 
            
        elif self.closure_duration >= self.DROWSY_TIME:
            self.state = "DROWSY (MICROSLEEP)"
            self.current_alert_code = 1
            self.state_color = (0, 255, 255) 
            
        else:
            self.state = "SAFE"
            self.current_alert_code = 0
            self.state_color = (0, 255, 0) 

        self._send_alert(str(self.current_alert_code))
        self._draw_ui(canvas, h, w, smoothed_ear)
        return canvas

    def _send_alert(self, command):
        if time.time() - self.last_serial_time > 0.05: # Send command quickly (20 times per second max)
            if self.arduino and self.arduino.is_open:
                try:
                    self.arduino.write(command.encode())
                    self.last_serial_time = time.time()
                except:
                    pass

    def _draw_ui(self, img, h, w, ear):
        cv2.rectangle(img, (0, 0), (w, 100), (20, 20, 20), -1)
        
        cv2.putText(img, f"STATUS: {self.state}", (30, 60), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, self.state_color, 2)
        
        cv2.putText(img, f"EAR: {ear:.3f}", (w - 280, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Closed Time: {self.closure_duration:.1f}s", (w - 280, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.current_alert_code == 2:
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 15)
            warn = "PULL OVER IMMEDIATELY"
            ts = cv2.getTextSize(warn, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)[0]
            cv2.putText(img, warn, ((w - ts[0]) // 2, h - 60), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)

    def _draw_calibration(self, img, h, w, progress):
        cv2.putText(img, "CALIBRATING EYES...", (w//4, h//2), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)
        cv2.rectangle(img, (w//4, h//2 + 20), 
                      (w//4 + int((w//2) * progress), h//2 + 40), (0, 255, 0), -1)

    def cleanup(self):
        """THE KILL SWITCH"""
        if self.arduino and self.arduino.is_open:
            print("\n🛑 SHUTTING DOWN ENGINE...")
            try:
                for _ in range(5):
                    self.arduino.write(b'3')
                    self.arduino.flush()
                    time.sleep(0.05)
            except:
                pass
            finally:
                self.arduino.close()
        self.face_mesh.close()

if __name__ == "__main__":
    port = connect_arduino("COM5")
    detector = DrowsinessDetector(port)
    video = WebcamVideoStream().start()
    
    print("\nSYSTEM ACTIVE - Press ESC to Exit")
    
    try:
        while True:
            frame = video.read()
            if frame is None: continue
            
            frame = cv2.flip(frame, 1)
            processed = detector.process_frame(frame)
            
            if processed is not None:
                cv2.imshow("Smart Drowsiness Monitor", processed)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        video.stop()
        cv2.destroyAllWindows()
        detector.cleanup() 
        print("Safely Terminated.")