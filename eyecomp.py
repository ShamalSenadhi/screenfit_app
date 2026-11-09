"""Browser-side eye capture (MediaPipe runs in the user's browser; only numbers reach Python, video never leaves the device).
Same interface as eyes.EyeProcessor: label, target, df(), received, errors, release()."""
import os, threading, numpy as np, pandas as pd, streamlit.components.v1 as components

_comp = components.declare_component("eye_cam", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_component"))
COLS = ["t", "label", "tx", "ty", "ok", "ear", "ix", "iy", "yaw", "pitch"]

class BrowserEyes:
    def __init__(self):
        self.rows, self.label, self.target, self.lock = [], "idle", (np.nan, np.nan), threading.Lock()
        self.ack, self.received, self.errors, self.playing, self.error, self.closed = 0, 0, 0, False, "", False

    def ingest(self, val):
        """Append new rows once (rows carry an index; Python acknowledges them so nothing is lost or duplicated)."""
        if not isinstance(val, dict) or self.closed: return
        self.playing, self.errors, self.error = bool(val.get("playing")), int(val.get("errors", 0) or 0), str(val.get("error") or "")
        with self.lock:
            for r in val.get("rows") or []:
                if r["i"] > self.ack:
                    self.rows.append({k: (np.nan if r.get(k) is None else r.get(k)) for k in COLS}); self.ack = r["i"]; self.received += 1

    def df(self):
        with self.lock: return pd.DataFrame(self.rows, columns=COLS) if self.rows else pd.DataFrame(columns=COLS)

    def release(self): self.closed = True

def render(proc, key="eyecam"):
    """Show the camera widget (call once per run) and ingest what the browser sent."""
    nn = lambda x: None if x != x else float(x)
    proc.ingest(_comp(label=proc.label, tx=nn(proc.target[0]), ty=nn(proc.target[1]), ack=proc.ack, key=key, default=None))
