"""Webcam processor (MediaPipe Face Mesh) for streamlit-webrtc. Keeps only numeric signals, never frames."""
import time, threading, numpy as np, pandas as pd
from streamlit_webrtc import VideoProcessorBase
import mediapipe as mp
R = [33, 160, 158, 133, 153, 144]; L = [362, 385, 387, 263, 373, 380]

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.rows, self.label, self.target, self.lock = [], "idle", (np.nan, np.nan), threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24"); res = self.fm.process(img)
        row = dict(t=time.time(), label=self.label, tx=self.target[0], ty=self.target[1], ok=0)
        if res.multi_face_landmarks:
            p = np.array([[l.x, l.y] for l in res.multi_face_landmarks[0].landmark]); n = np.linalg.norm
            ear = lambda i: (n(p[i[1]] - p[i[5]]) + n(p[i[2]] - p[i[4]])) / (2 * n(p[i[0]] - p[i[3]]))
            w = n(p[234] - p[454])
            row.update(ok=1, ear=(ear(R) + ear(L)) / 2, ix=(p[468, 0] - p[33, 0]) / (p[133, 0] - p[33, 0]),
                       iy=(p[468, 1] - p[159, 1]) / (p[145, 1] - p[159, 1]),
                       yaw=(p[1, 0] - (p[234, 0] + p[454, 0]) / 2) / w, pitch=(p[1, 1] - p[152, 1]) / n(p[10] - p[152]))
        with self.lock: self.rows.append(row)
        return frame

    def df(self):
        with self.lock: return pd.DataFrame(self.rows)
