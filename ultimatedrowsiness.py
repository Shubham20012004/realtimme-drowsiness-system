import cv2
import mediapipe as mp
import serial
import time
import numpy as np
from collections import deque
import json
import sqlite3
import logging
from datetime import datetime
import threading
import sys
import glob
from queue import Queue, Empty
 

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("system_events.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def find_arduino_port(default_port="COM5"):
    ports_to_try = [default_port]
    if sys.platform.startswith('win'):
        ports_to_try += [f'COM{i}' for i in range(1, 256) if f'COM{i}' != default_port]
    elif sys.platform.startswith('darwin'):
        ports_to_try += glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/cu.usbserial*')
    else:
        ports_to_try += glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')

    for port in ports_to_try:
        try:
            s = serial.Serial(port, timeout=0.1, write_timeout=0.1)
            s.close()
            return port
        except (OSError, serial.SerialException):
            pass
    return None


class AsyncTelemetryLogger:
    def __init__(self, db_name="drowsiness_telemetry.db"):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db_name = db_name
        self.queue = Queue()
        self.frame_count = 0
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _initialize_database(self, conn, cursor):
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, timestamp TEXT, frame_number INTEGER,
                ear REAL, mar REAL, perclos REAL, pitch REAL, yaw REAL, roll REAL,
                blink_rate REAL, microsleep_events INTEGER, fatigue_score INTEGER,
                confidence REAL, system_state TEXT
            )
        ''')
        conn.commit()

    def _process_queue(self):
        conn = sqlite3.connect(self.db_name, check_same_thread=False)
        cursor = conn.cursor()
        self._initialize_database(conn, cursor)
        json_file = open(f"session_{self.session_id}.jsonl", "w", encoding='utf-8')
        
        db_batch = []
        BATCH_SIZE = 30
        
        while True:
            try:
                metrics = self.queue.get(timeout=1.0)
                if metrics is None: break
                
                json_file.write(json.dumps(metrics) + "\n")
                
                m = metrics['metrics']
                db_batch.append((
                    metrics['session_id'], metrics['timestamp'], metrics['frame'],
                    m.get('ear', 0), m.get('mar', 0), m.get('perclos', 0),
                    m.get('pitch', 0), m.get('yaw', 0), m.get('roll', 0),
                    m.get('blink_rate', 0), m.get('microsleep', 0),
                    m.get('fatigue_score', 0), m.get('confidence', 0), m.get('state', 'UNKNOWN')
                ))

                if len(db_batch) >= BATCH_SIZE:
                    cursor.executemany('''
                        INSERT INTO telemetry (session_id, timestamp, frame_number, ear, mar, perclos, 
                                               pitch, yaw, roll, blink_rate, microsleep_events, 
                                               fatigue_score, confidence, system_state)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', db_batch)
                    conn.commit()
                    db_batch.clear()
                    
            except Empty: continue
            except Exception as e: logging.error(f"Telemetry Error: {e}")
                
        if db_batch:
            cursor.executemany('''INSERT INTO telemetry
                (session_id, timestamp, frame_number, ear, mar, perclos, pitch, yaw, roll,
                 blink_rate, microsleep_events, fatigue_score, confidence, system_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', db_batch)
            conn.commit()
        conn.close()
        json_file.close()

    def log_frame(self, metrics):
        self.frame_count += 1
        self.queue.put({
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "frame": self.frame_count,
            "metrics": metrics
        })

    def cleanup(self):
        self.queue.put(None)
        self.worker.join()


class WebcamVideoStream:
    def __init__(self, src=0, width=1280, height=720):
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, backend)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
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
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.003)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()


class BlinkAnalyzer:
    def __init__(self):
        self.blink_history = deque(maxlen=100)
        self.last_blink_time = 0
        self.microsleep_count = 0
        
    def analyze_blink(self, ear, threshold, current_time):
        if ear < threshold:
            if not hasattr(self, 'blink_start'): self.blink_start = current_time
            return None
        else:
            if hasattr(self, 'blink_start'):
                duration = current_time - self.blink_start
                interval = current_time - self.last_blink_time
                self.blink_history.append({'duration': duration, 'interval': interval, 'timestamp': current_time})
                self.last_blink_time = current_time
                delattr(self, 'blink_start')
                if duration > 0.8: 
                    self.microsleep_count += 1
                    return 'MICROSLEEP'
                return 'NORMAL'
            return None
    
    def get_blink_rate(self):
        if len(self.blink_history) < 2: return 0
        recent = [b for b in self.blink_history if b['duration'] < 0.8]
        if len(recent) < 2: return 0
        span = recent[-1]['timestamp'] - recent[0]['timestamp']
        if span == 0: return 0
        return (len(recent) / span) * 60


class UltimateDrowsinessDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]
    
    CALIB_FRAMES = 150 
    PERCLOS_WINDOW = 90
    
   
    DROWSY_TIME = 0.5    
    CRITICAL_TIME = 1.5  
    COOLDOWN_TIME = 4.0  
    
    def __init__(self, default_baud=9600):
        self.logger = AsyncTelemetryLogger()
        self.baud_rate = default_baud
        self.arduino = self._connect_serial()
        self.blink_analyzer = BlinkAnalyzer()
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        self.ear_baseline = None
        self.mar_baseline = None
        self.pitch_baseline = None
        self.calib_data = {'ear': [], 'mar': [], 'pitch': []}
        
        self.ear_buffer = deque(maxlen=7)
        self.mar_buffer = deque(maxlen=7)
        self.perclos_buffer = deque(maxlen=self.PERCLOS_WINDOW)
        
        self.closure_start_time = None
        self.closure_duration = 0.0
        self.cooldown_active = False
        self.cooldown_end_time = 0.0
        
        self.state = "CALIBRATING"
        self.state_color = (0, 255, 0)
        self.current_alert_code = 0
        self.last_alert_time = 0

    def _connect_serial(self):
        port = find_arduino_port()
        if port:
            try:
                conn = serial.Serial(port, self.baud_rate, timeout=0.1, write_timeout=0.1)
                time.sleep(2)
                logging.info(f"✓ Hardware active on {port}")
                return conn
            except Exception as e:
                logging.warning(f"⚠ Serial connection failed: {e}")
        logging.warning("⚠ Operating in CV-only mode. Audio fallback enabled.")
        return None

    @staticmethod
    def euclidean_distance_3d(p1, p2): 
        
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def calculate_ear_3d(self, eye_points):
        A = self.euclidean_distance_3d(eye_points[1], eye_points[5])
        B = self.euclidean_distance_3d(eye_points[2], eye_points[4])
        C = self.euclidean_distance_3d(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C) if C != 0 else 0

    def calculate_mar_3d(self, mouth_points):
        A = self.euclidean_distance_3d(mouth_points[2], mouth_points[10])
        B = self.euclidean_distance_3d(mouth_points[4], mouth_points[8])
        C = self.euclidean_distance_3d(mouth_points[0], mouth_points[6])
        return (A + B) / (2.0 * C) if C != 0 else 0

    def calculate_head_pose(self, landmarks, w, h):
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
        image_points = np.array([(landmarks[1].x*w, landmarks[1].y*h), (landmarks[152].x*w, landmarks[152].y*h), (landmarks[33].x*w, landmarks[33].y*h), (landmarks[263].x*w, landmarks[263].y*h), (landmarks[61].x*w, landmarks[61].y*h), (landmarks[291].x*w, landmarks[291].y*h)], dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64), np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0, 0, 0
        pose_mat = cv2.hconcat((cv2.Rodrigues(rot_vec)[0], trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        return euler[0][0], euler[1][0], euler[2][0]

    def process_frame(self, frame):
        if frame is None: return None
        h, w, _ = frame.shape
        canvas = frame.copy()
        current_time = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            self._handle_no_face(canvas)
            return canvas

        landmarks = results.multi_face_landmarks[0].landmark
        
       
        coords_2d = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks], dtype=np.int32)
        
        
        coords_3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float64)
        
        raw_ear = (self.calculate_ear_3d(coords_3d[self.LEFT_EYE]) + self.calculate_ear_3d(coords_3d[self.RIGHT_EYE])) / 2
        raw_mar = self.calculate_mar_3d(coords_3d[self.MOUTH_INNER])
        pitch, yaw, roll = self.calculate_head_pose(landmarks, w, h)
        
      
        if self.ear_baseline is None:
            self.calib_data['ear'].append(raw_ear)
            self.calib_data['mar'].append(raw_mar)
            self.calib_data['pitch'].append(pitch)
            progress = len(self.calib_data['ear']) / self.CALIB_FRAMES
            self._render_calibration_ui(canvas, progress)
            if len(self.calib_data['ear']) >= self.CALIB_FRAMES:
               
                eyes_open_ear = np.percentile(self.calib_data['ear'], 90)
                
                
                self.ear_baseline = eyes_open_ear * 0.78 
                
                self.mar_baseline = np.percentile(self.calib_data['mar'], 75) * 1.25
                self.pitch_baseline = np.mean(self.calib_data['pitch']) 
                self.state = "SAFE"
            return canvas
        
       
        self.ear_buffer.append(raw_ear)
        self.mar_buffer.append(raw_mar)
        ear_final = np.median(list(self.ear_buffer))
        mar_final = np.median(list(self.mar_buffer))

       
        if ear_final < self.ear_baseline:
            if self.closure_start_time is None:
                self.closure_start_time = current_time
            self.closure_duration = current_time - self.closure_start_time
        else:
            self.closure_start_time = None
            self.closure_duration = 0.0

       
        blink_rate = self.blink_analyzer.get_blink_rate()
        self.blink_analyzer.analyze_blink(ear_final, self.ear_baseline, current_time)
        self.perclos_buffer.append(1 if ear_final < self.ear_baseline else 0)
        perclos = sum(self.perclos_buffer) / len(self.perclos_buffer)
        
        fatigue = 0
        if perclos > 0.35: fatigue += 3
        if blink_rate < 8 or blink_rate > 35: fatigue += 2
        if mar_final > self.mar_baseline: fatigue += 2
        if pitch < (self.pitch_baseline - 15.0): fatigue += 3

      
        is_critical_event = self.closure_duration >= self.CRITICAL_TIME
        
        if is_critical_event:
            self.current_alert_code = 2
            self.state = "CRITICAL (SLEEPING)"
            self.state_color = (0, 0, 255)
            self.cooldown_active = True
            self.cooldown_end_time = current_time + self.COOLDOWN_TIME
            
        elif self.cooldown_active:
            if current_time < self.cooldown_end_time:
                self.current_alert_code = 2
                remaining = self.cooldown_end_time - current_time
                self.state = f"COOLDOWN ({remaining:.1f}s)"
                self.state_color = (0, 165, 255) 
            else:
                self.cooldown_active = False 
                self.closure_duration = 0.0 
                
        elif self.closure_duration >= self.DROWSY_TIME:
            self.current_alert_code = 1
            self.state = "DROWSY (MICROSLEEP)"
            self.state_color = (0, 255, 255)
            
        else:
            self.current_alert_code = 0
            self.state = "SAFE"
            self.state_color = (0, 255, 0)
        
        self._send_alert(self.current_alert_code)
        
        self.logger.log_frame({
            'ear': ear_final, 'mar': mar_final, 'perclos': perclos,
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'blink_rate': blink_rate, 'microsleep': self.blink_analyzer.microsleep_count,
            'fatigue_score': fatigue, 'state': self.state
        })
        
        self._render_advanced_hud(canvas, ear_final, mar_final, pitch, blink_rate, fatigue, coords_2d)
        return canvas

    def _handle_no_face(self, canvas):
        self.state = "NO FACE DETECTED"
        self.state_color = (0, 0, 255)
        self.current_alert_code = 2 
        self._send_alert(self.current_alert_code)
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 100), -1)
        cv2.putText(canvas, "NO FACE DETECTED", (w//2 - 200, h//2), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 100, 255), 3)

    def _send_alert(self, level):
        current_time = time.time()
        if current_time - self.last_alert_time < 0.1: return
        
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(str(level).encode())
                self.last_alert_time = current_time
            except serial.SerialException:
                logging.warning("⚠ Serial connection lost! Attempting auto-reconnect...")
                self.arduino.close()
                self.arduino = self._connect_serial()
        else:
            if level == 2 and HAS_WINSOUND:
                winsound.Beep(2000, 200)

    def _render_advanced_hud(self, frame, ear, mar, pitch, blink_rate, fatigue, coords):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (20, 20), (450, 90), self.state_color, 3)
        cv2.putText(frame, f"STATUS: {self.state}", (35, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, self.state_color, 2)
        
        cv2.rectangle(frame, (w - 320, 20), (w - 20, 250), (30, 30, 30), -1)
        texts = [
            f"3D EAR: {ear:.3f} (Base: {self.ear_baseline:.3f})",
            f"Eye Closed: {self.closure_duration:.2f}s",
            f"Pitch Base: {self.pitch_baseline:.1f}",
            f"Blink Rate: {blink_rate:.1f}/min",
            f"Fatigue Score: {fatigue}"
        ]
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (w - 305, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.polylines(frame, [coords[self.LEFT_EYE]], True, (0, 255, 255), 1)
        cv2.polylines(frame, [coords[self.RIGHT_EYE]], True, (0, 255, 255), 1)
        cv2.polylines(frame, [coords[self.MOUTH_INNER]], True, (255, 200, 0), 1)
        
        if self.current_alert_code == 2:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)

    def _render_calibration_ui(self, frame, progress):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"CALIBRATING 3D SPACE: {int(progress * 100)}%", (w//2 - 250, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    def cleanup(self):
        if self.arduino and self.arduino.is_open:
            try:
                for _ in range(3):
                    self.arduino.write(b'3')
                    time.sleep(0.05)
            except: pass
            finally: self.arduino.close()
        self.face_mesh.close()
        self.logger.cleanup() 


if __name__ == "__main__":
    detector = UltimateDrowsinessDetector()
    video_stream = WebcamVideoStream(src=0, width=1280, height=720).start()
    time.sleep(2.0)
    
    try:
        while True:
            frame = video_stream.read()
            if frame is None: continue
            
            processed = detector.process_frame(cv2.flip(frame, 1))
            cv2.imshow("God Level Drowsiness System", processed)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt: pass
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()
        detector.cleanup()
