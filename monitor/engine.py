"""Causal measurements for stationary research sessions. All timestamps are seconds."""
from collections import deque
from math import dist, isfinite
from statistics import median

LEFT = (33, 160, 158, 133, 153, 144)
RIGHT = (362, 385, 387, 263, 373, 380)
SCHEMA = "geometry-time-v1"
FEATURES = ["left", "right", "mar", "normalized", "closure_s", "perclos", "pitch", "yaw", "roll", "coverage"]


def ear(points):
    width = dist(points[0], points[3])
    if width <= 0:
        raise ValueError("Coincident eye corners")
    return (dist(points[1], points[5]) + dist(points[2], points[4])) / (2 * width)


class Engine:
    def __init__(self, calibration_s=5, window=30, max_gap=0.3):
        if calibration_s<=0 or window<=0 or max_gap<=0:
            raise ValueError("Calibration duration, window and maximum gap must be positive")
        self.calibration_s, self.window, self.max_gap = calibration_s, window, max_gap
        self.last = self.start = self.segment = self.closed_since = None
        self.samples = deque(maxlen=4096)
        self.intervals = deque(maxlen=16384)
        self.blinks = deque(maxlen=1024)
        self.baseline = self.previous = None
        self.alert_state = None
        self.recovery_since = None
        self.yawn_start = self.nod_start = self.last_pitch = None
        self.yawns = deque(maxlen=1024)
        self.nods = deque(maxlen=1024)
        self.current_yawn_counted = self.current_nod_counted = False

    def update(self, t, obs, reason="TRACKING_LOST"):
        if not isfinite(t) or (self.last is not None and t <= self.last):
            raise ValueError("Frame time must increase strictly")
        dt = t - self.last if self.last is not None else 0
        self.last = t
        if self.start is None:
            self.start = t
        out = dict(t=t, schema=SCHEMA, status=reason, state=None)
        if obs is None or dt > self.max_gap:
            self.previous = self.closed_since = self.segment = None
            self.alert_state = self.recovery_since = None
            self.samples.clear()
            self.yawn_start = self.nod_start = self.last_pitch = None
            self.current_yawn_counted = self.current_nod_counted = False
        if self.baseline is None and t-self.start>max(20,4*self.calibration_s):
            self.start=t
            self.segment=None
            self.samples.clear()
            return dict(out,status="CALIBRATION_TIMEOUT",reason="Insufficient contiguous valid observations")
        if obs is None:
            return out
        out.update(obs)
        if self.baseline is None:
            if self.segment is None:
                self.segment = t
            self.samples.append(tuple(obs[k] for k in ("left", "right", "mar", "pitch", "yaw", "roll")))
            elapsed = t - self.segment
            out.update(status="CALIBRATING", progress=min(1, elapsed/self.calibration_s))
            if elapsed < self.calibration_s or len(self.samples) < 12:
                return out
            columns = list(zip(*self.samples))
            centers = [median(c) for c in columns]
            deviations = [median(abs(x-m) for x in c) for c, m in zip(columns, centers)]
            if min(centers[:2]) < 0.13 or max(deviations[:2]) > 0.035 or max(deviations[3:]) > 4:
                self.samples.clear()
                self.segment = None
                out.update(status="CALIBRATION_RETRY", reason="Unstable pose or eye opening")
                return out
            self.baseline = dict(zip(("left","right","mar","pitch","yaw","roll"), centers))
            self.samples.clear()
            out["calibration_mad"]=dict(zip(("left","right","mar","pitch","yaw","roll"),deviations))
            out["calibration_s"]=elapsed
        normalized = (obs["left"]/self.baseline["left"] + obs["right"]/self.baseline["right"])/2
        closed = normalized < 0.72
        if self.previous is not None and dt <= self.max_gap:
            self.intervals.append((t-dt, t, self.previous and closed))
        if closed and self.closed_since is None:
            self.closed_since = t
        if not closed and self.closed_since is not None:
            self.blinks.append((t, t-self.closed_since))
            self.closed_since = None
        cutoff = t-self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        while self.blinks and self.blinks[0][0] <= cutoff:
            self.blinks.popleft()
        valid = sum(b-max(a,cutoff) for a,b,_ in self.intervals)
        shut = sum(b-max(a,cutoff) for a,b,c in self.intervals if c)
        closure = 0 if self.closed_since is None else t-self.closed_since
        for history in (self.yawns,self.nods):
            while history and history[0] <= cutoff:
                history.popleft()
        if obs["mar"] > max(.35,self.baseline["mar"]+.20):
            if self.yawn_start is None:
                self.yawn_start=t
            if t-self.yawn_start>=1.0 and not self.current_yawn_counted:
                self.yawns.append(t)
                self.current_yawn_counted=True
        else:
            self.yawn_start=None
            self.current_yawn_counted=False
        if obs["pitch"]-self.baseline["pitch"]>12:
            if self.nod_start is None:
                self.nod_start=t
            if t-self.nod_start>=.25 and not self.current_nod_counted:
                self.nods.append(t)
                self.current_nod_counted=True
        else:
            self.nod_start=None
            self.current_nod_counted=False
        velocity=None if self.last_pitch is None or not dt else (obs["pitch"]-self.last_pitch)/dt
        self.last_pitch=obs["pitch"]
        self.previous = closed
        durations = [d for _,d in self.blinks if 0.06 <= d < 0.8]
        coverage = min(1,valid/self.window)
        candidate = "CRITICAL" if closure >= 1.5 else "DROWSY" if closure >= 0.5 else "ALERT"
        severity = {"ALERT":0,"DROWSY":1,"CRITICAL":2}
        if self.alert_state is None or severity[candidate]>=severity[self.alert_state]:
            self.alert_state=candidate
            self.recovery_since=None
        elif self.recovery_since is None:
            self.recovery_since=t
        elif t-self.recovery_since>=.25:
            self.alert_state=candidate
            self.recovery_since=None
        out.update(status="RUNNING", state=self.alert_state,
                   baseline=self.baseline, normalized=normalized, closure_s=closure, coverage=coverage,
                   perclos=shut/valid if coverage>=0.7 else None,
                   blink_per_min=60*len(durations)/valid if coverage>=0.7 else None,
                   yawn_s=t-self.yawn_start if self.yawn_start is not None else 0,
                   yawns_per_min=60*len(self.yawns)/valid if coverage>=.7 else None,
                   nods_per_min=60*len(self.nods)/valid if coverage>=.7 else None,
                   pitch_velocity_deg_s=velocity,
                   max_closure_s=max([closure]+[duration for _,duration in self.blinks]),
                   prolonged_closures=sum(duration>=.8 for _,duration in self.blinks),
                   mean_blink_s=sum(durations)/len(durations) if durations else None)
        return out
