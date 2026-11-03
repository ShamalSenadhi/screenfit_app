"""Offline tests (no Ollama, no webcam):  python tests/test_pipeline.py   (run from the project folder)"""
import sys, os, json, random, shutil, tempfile
sys.path.insert(0, os.getcwd()); os.makedirs("data/sessions", exist_ok=True)
import numpy as np, pandas as pd
import llm, bank, features, report

def fake_ask(system, user, temperature=0.0, model=None):
    if "Grade ONLY" in system or "examiner" in system:
        return {"correctness": 2, "completeness": 1, "justification": "Mentions the key idea but not the trade-off."}
    area = user.split(".")[0]
    return {"paragraph": " ".join(["word"] * 70), "questions": [
        {"type": "mcq", "text": f"{area} Q1", "options": ["a", "b", "c", "d"], "answer": 1},
        {"type": "mcq", "text": f"{area} Q2", "options": ["a", "b", "c", "d"], "answer": 2},
        {"type": "tf", "text": f"{area} Q3", "answer": True},
        {"type": "short", "text": f"{area} Q4", "key": "key one", "rubric": "r"},
        {"type": "short", "text": f"{area} Q5", "key": "key two", "rubric": "r"}]}
llm.ask = fake_ask; bank.ask = fake_ask

# 1) bank generation + assembly --------------------------------------------------
b = []
for c, areas in bank.CLUSTERS.items():
    for a in areas:
        it = bank.make_item(a, c); assert it, a; it["approved"] = True; b.append(it)
assert len(b) == 23
for seed in range(50):
    items = bank.assemble(b, seed)
    assert len(items) == 5 and all(len(i["questions"]) == 5 for i in items)
    assert items[0]["cluster"] in ("Programming", "Data and APIs")
    assert [i["cluster"] for i in items[1:]] == ["Data science, AI and ML", "Web and security", "Systems", "Engineering practice"]
    assert all(q["options"][q["answer"]] for i in items for q in i["questions"][:2])
assert bank.assemble(b, 7) == bank.assemble(b, 7)          # reproducible from the seed
print("bank OK: 23 areas, 50 random sets valid and reproducible")

# 2) LLM grading (mocked) --------------------------------------------------------
g = llm.grade_short({"text": "t", "key": "k", "rubric": "r"}, "my answer", runs=3)
assert abs(g["score"] - 0.75) < 1e-9 and g["agree"], g
assert llm.grade_short({"text": "t", "key": "k"}, "  ")["score"] == 0.0
print("grading OK:", g["score"])

# 3) eye features from synthetic signals ---------------------------------------
rng = np.random.default_rng(1); rows = []; t = 1000.0; FPS = 30
def seg(label, secs, blink_every, tgt=None, scan=False, ear0=0.30):
    global t
    for n in range(int(secs * FPS)):
        ear = ear0 + 0.006 * rng.standard_normal(); ph = (n / FPS) % blink_every
        if ph < 0.15: ear = 0.07 + 0.01 * rng.standard_normal()
        sx, sy = ((n / FPS * 0.35) % 1.0, 0.3 + 0.1 * int(n / FPS / 3)) if scan else (tgt or (0.5, 0.5))
        rows.append(dict(t=t, label=label, tx=tgt[0] if tgt else np.nan, ty=tgt[1] if tgt else np.nan, ok=1, ear=ear,
                         ix=0.3 + 0.4 * sx + 0.004 * rng.standard_normal(), iy=0.3 + 0.4 * sy + 0.004 * rng.standard_normal(),
                         yaw=0.1 * sx - 0.05, pitch=0.2 * sy + 0.1)); t += 1 / FPS
seg("setup", 5, 4)
for k, tg in enumerate([(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]): seg("calib", 10, 4, tgt=tg)
seg("baseline", 60, 4)                                       # 15 blinks/min
for p in range(1, 6):
    seg(f"read{p}", 35, 4 + p, scan=True); seg(f"q{p}", 60, 4 + p)   # blinking slows over the session
df = pd.DataFrame(rows); f = features.compute(df)
assert f is not None and f["quality"]["gaze_err_deg"] < 3, f["quality"]
b0 = f["baseline"]["blink_rate"]; br = f["para"].blink_rate
assert 12 < b0 < 18 and br.iloc[-1] < br.iloc[0], (b0, br.tolist())
print(f"features OK: baseline blink rate {b0:.1f}/min, paragraphs {br.round(1).tolist()}, gaze error {f['quality']['gaze_err_deg']:.2f} deg")
print(f["para"].round(1).to_string(index=False))

# 4) scoring + dashboard record ------------------------------------------------
items = bank.assemble(b, 3); ans = {}; grades = {}
for pi in range(5):
    for qi, q in enumerate(items[pi]["questions"]):
        k = f"{pi}-{qi}"
        if q["type"] == "mcq": ans[k] = dict(val=q["answer"] if qi == 0 else (q["answer"] + 1) % 4, timed_out=False)
        elif q["type"] == "tf": ans[k] = dict(val=True, timed_out=False)
        else: ans[k] = dict(val="some answer", timed_out=False); grades[k] = llm.grade_short(q, "some answer")
meta = dict(level="Mid-level", glasses=True, screen_hours=9, cvsq=7, osdi=19)
rep = report.build("test0001", 3, meta, items, ans, grades, f, dict(tlx=48, strain=4))
assert rep["visual"]["risk"] == "Moderate" and 0 <= rep["technical"]["overall"] <= 100
assert report.risk_level(2, 5) == "Low" and report.risk_level(8, 30) == "High" and report.risk_level(6, 10) == "Moderate"
json.dump(rep, open("data/sessions/test0001.json", "w")); json.loads(open("data/sessions/test0001.json").read())
print("report OK: score", round(rep["technical"]["overall"]), "risk", rep["visual"]["risk"], "flags", rep["visual"]["flags"])

# 5) Streamlit pages via AppTest --------------------------------------------------
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run(); assert not at.exception, at.exception
at.sidebar.radio[0].set_value("Recruiter dashboard").run(); assert not at.exception, at.exception
assert any("Technical score" in m.label for m in at.metric), [m.label for m in at.metric]
at.sidebar.radio[0].set_value("Item bank (admin)").run(); assert not at.exception, at.exception
print("Streamlit pages OK: candidate form, recruiter dashboard, admin")

# 6) webcam processor on a blank frame -------------------------------------------
import av, eyes
p = eyes.EyeProcessor(); fr = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), np.uint8), format="rgb24"); p.label = "setup"; p.recv(fr)
assert len(p.df()) == 1 and p.df().ok.iloc[0] == 0
print("EyeProcessor OK (no face in blank frame)")
os.remove("data/sessions/test0001.json"); print("ALL TESTS PASSED")
