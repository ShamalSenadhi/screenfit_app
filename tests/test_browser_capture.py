"""Browser capture: JS geometry == Python formulas, ack protocol, features, and the app flow:  python tests/test_browser_capture.py"""
import sys, os, json, subprocess, shutil; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, features, eyecomp

# 1) JS geometry matches the Python formulas (random landmarks) ----------------------------------
rng = np.random.default_rng(5); P = rng.random((478, 2)); R_, L_ = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]; n = np.linalg.norm
ear = lambda i: (n(P[i[1]] - P[i[5]]) + n(P[i[2]] - P[i[4]])) / (2 * n(P[i[0]] - P[i[3]])); w = n(P[234] - P[454])
want = dict(ear=(ear(R_) + ear(L_)) / 2, ix=(P[468, 0] - P[33, 0]) / (P[133, 0] - P[33, 0]), iy=(P[468, 1] - P[159, 1]) / (P[145, 1] - P[159, 1]),
            yaw=(P[1, 0] - (P[234, 0] + P[454, 0]) / 2) / w, pitch=(P[1, 1] - P[152, 1]) / n(P[10] - P[152]))
if shutil.which("node"):
    js = "const g=require('./eye_component/geometry.js');const p=JSON.parse(process.argv[1]).map(([x,y])=>({x,y}));console.log(JSON.stringify(g.row(p)))"
    got = json.loads(subprocess.run(["node", "-e", js, json.dumps(P.tolist())], capture_output=True, text=True, check=True).stdout)
    assert all(abs(got[k] - want[k]) < 1e-9 for k in want) and got["ok"] == 1, (got, want); print("JS geometry matches Python formulas")
html = open("eye_component/index.html").read()
for needle in ("streamlit:componentReady", "streamlit:setComponentValue", "streamlit:render", "getUserMedia", "detectForVideo", "geometry.js"): assert needle in html, needle

# 2) ack protocol: nothing lost, nothing duplicated, even if batches repeat or arrive late ------------
p = eyecomp.BrowserEyes(); mk = lambda a, b: [dict(i=i, t=1000 + i / 30, label="read1", tx=None, ty=None, ok=1, ear=0.3, ix=.5, iy=.5, yaw=0, pitch=0) for i in range(a, b)]
p.ingest(dict(playing=True, errors=0, error="", rows=mk(1, 31))); assert p.ack == 30 and p.received == 30 and p.playing
p.ingest(dict(playing=True, rows=mk(1, 31)))                       # same batch again (browser had not seen the ack yet)
p.ingest(dict(playing=True, rows=mk(20, 61)))                      # overlapping batch
assert p.ack == 60 and len(p.df()) == 60 and p.df().t.is_monotonic_increasing and p.df().tx.isna().all()
p.ingest(dict(playing=False, error="NotAllowedError: Permission denied", rows=[])); assert p.error.startswith("NotAllowed") and not p.playing
p.release(); p.ingest(dict(playing=True, rows=mk(61, 70))); assert len(p.df()) == 60
print("ack protocol OK: no loss, no duplicates; errors surfaced")

# 3) rows from the browser go through the same feature pipeline --------------------------------------
rows = []; t = 5000.0
def seg(label, secs, every, ear0=0.30):
    global t
    for k in range(int(secs * 30)):
        e = ear0 + 0.006 * rng.standard_normal(); e = 0.07 if (k / 30) % every < 0.15 else e
        rows.append(dict(i=len(rows) + 1, t=t, label=label, tx=None, ty=None, ok=1, ear=e, ix=.5, iy=.5, yaw=0, pitch=0)); t += 1 / 30
seg("baseline", 60, 4)
for q in range(1, 6): seg(f"read{q}", 35, 4 + q); seg(f"q{q}", 60, 4 + q)
b = eyecomp.BrowserEyes()
for a in range(0, len(rows), 400): b.ingest(dict(playing=True, rows=rows[a:a + 400]))
f = features.compute(b.df()); assert f and 12 < f["baseline"]["blink_rate"] < 18 and f["frames"]["total"] == len(rows) and len(f["events"]) > 50
print(f"features from browser rows OK: {f['frames']['total']} frames, baseline {f['baseline']['blink_rate']:.1f} blinks/min")

# 4) the app uses it by default (camera page enables the interview button when the browser reports a face) -----------
import semantic; semantic._M.update(m=None, err="offline")
from streamlit.testing.v1 import AppTest
def fake_render(proc, key="eyecam"):
    proc.playing = True; now = 9000.0 + len(proc.rows) / 30
    proc.ingest(dict(playing=True, errors=0, error="", rows=[dict(i=proc.ack + k + 1, t=now + k / 30, label=proc.label, tx=None, ty=None, ok=1, ear=.3, ix=.5, iy=.5, yaw=0, pitch=0) for k in range(60)]))
eyecomp.render = fake_render
at = AppTest.from_file("app.py", default_timeout=60).run(); at.checkbox[0].check(); at.run()
[x for x in at.button if "Continue" in x.label][0].click().run(); assert not at.exception, at.exception
assert at.session_state["stage"] == "camera" and at.radio[0].value == "browser"
begin = [x for x in at.button if "Begin" in x.label][0]; assert not begin.disabled, "interview button should be enabled once the browser reports a face"
begin.click().run(); assert not at.exception, at.exception and at.session_state["stage"] == "run"
print("app flow OK with browser capture (default)"); print("PASSED")
