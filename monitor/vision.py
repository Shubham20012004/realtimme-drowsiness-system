import math

import cv2
import mediapipe as mp
import numpy as np

from .engine import LEFT, RIGHT, ear

MODEL = np.array([(0,0,0),(0,63,-12),(-43,-32,-26),(43,-32,-26),(-28,28,-24),(28,28,-24)], dtype=float)


class Vision:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    def close(self):
        self.mesh.close()

    def observe(self, frame):
        h,w = frame.shape[:2]
        result = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None, "TRACKING_LOST"
        xy = np.array([(p.x*w, p.y*h) for p in result.multi_face_landmarks[0].landmark])
        key = xy[list(LEFT+RIGHT+(13,14,61,291,1,152))]
        if not np.isfinite(key).all() or (key<0).any() or (key >= [w,h]).any():
            return None, "TRACKING_DEGRADED"
        x,y,bw,bh = cv2.boundingRect(xy.astype(np.float32))
        roi = frame[max(0,y):min(h,y+bh),max(0,x):min(w,x+bw)]
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        light, blur = float(gray.mean()), float(cv2.Laplacian(gray,cv2.CV_64F).var())
        mouth_width = math.dist(xy[61],xy[291])
        if not 22 <= light <= 240 or blur < 8 or mouth_width < 8:
            return None, "TRACKING_DEGRADED"
        corners = xy[[1,152,33,263,61,291]]
        k = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=float)
        ok,rv,tv = cv2.solvePnP(MODEL,corners,k,None)
        if not ok:
            return None, "TRACKING_DEGRADED"
        projected = cv2.projectPoints(MODEL,rv,tv,k,None)[0].reshape(-1,2)
        residual = float(np.linalg.norm(projected-corners,axis=1).mean()/mouth_width)
        pitch,yaw,roll = cv2.RQDecomp3x3(cv2.Rodrigues(rv)[0])[0]
        if residual > 0.3 or abs(yaw)>50 or abs(pitch)>50:
            return None, "TRACKING_DEGRADED"
        try:
            left,right = (ear(xy[list(indices)]) for indices in (LEFT,RIGHT))
        except ValueError:
            return None, "TRACKING_DEGRADED"
        return dict(left=left,right=right,mar=math.dist(xy[13],xy[14])/mouth_width,
                    pitch=pitch,yaw=yaw,roll=roll,brightness=light,blur=blur,
                    reprojection_error=residual), "TRACKING_VALID"
