"""Exercises every interview step with a fake camera (no webcam):  python tests/test_run_stage.py"""
import sys, os, re, time, types
sys.path.insert(0, os.getcwd()); os.makedirs("data/sessions", exist_ok=True)
import numpy as np, pandas as pd, random
import llm, bank, semantic
semantic._M.update(m=None, err="offline")      # skip the real model download in tests
os.environ["GROQ_API_KEY"] = "test"   # grading key for the app check
exec(open("tests/test_pipeline.py").read().split("# 1) bank")[0].split("import llm, bank, features, report")[1])  # reuse fake_ask
src = open("app.py").read()
src = re.sub(r"def cam\(\):.*?return ctx, ctx.video_processor\n", '''def cam():
    class P:
        label = "idle"; target = (float("nan"), float("nan"))
        def df(self): return pd.DataFrame(dict(t=[time.time()] * 50, label=["x"] * 50, ok=[1] * 50))
    class C: state = types.SimpleNamespace(playing=True)
    return C(), P()
''', src, flags=re.S).replace("import bank, llm, features, report", "import types, bank, llm, features, report")
open("tests/_app_fake.py", "w").write(src)
from streamlit.testing.v1 import AppTest
b = []
for c, areas in bank.CLUSTERS.items():
    for a in areas:
        it = bank.make_item(a, c); it["approved"] = True; b.append(it)
bank.save(b)
at = AppTest.from_file("tests/_app_fake.py", default_timeout=60)
at.session_state["model"] = "x"; at.run(); assert not at.exception
# pre-session form -> camera stage
at.checkbox[0].check(); at.run()
[bt for bt in at.button if "Continue" in bt.label][0].click().run(); assert not at.exception, at.exception
assert at.session_state["stage"] == "camera", at.session_state["stage"]
[bt for bt in at.button if "Begin" in bt.label][0].click().run(); assert at.session_state["stage"] == "run"
steps = at.session_state["steps"]; want = 30 + 9 * 7 + 45 + 5 * (35 + 20 + 20 + 20 + 35 + 35) + 105 + 60
assert sum(s["dur"] for s in steps) == want == 1128, sum(s["dur"] for s in steps)
print("steps:", len(steps), "total", sum(s["dur"] for s in steps), "s; setup/calibration/baseline:", [s["dur"] for s in steps[:3]], "; question timers:", [s["dur"] for s in steps if s["k"] == "q"][:5])
# walk every step: questions advance ONLY with the Submit / Skip buttons (or the time-out); grading happens after the interview
click = lambda name: [b for b in at.button if name in b.label][0].click().run()
seen = set(); guard = 0; checked_empty = False; n_skip = n_timeout = n_sub = 0
while at.session_state["stage"] == "run" and guard < 90:
    guard += 1; i = at.session_state["i"]; k = steps[i]["k"]; seen.add(k); assert not at.exception, (k, at.exception)
    if k == "q":
        q = at.session_state["qs"][steps[i]["p"]]["questions"][steps[i]["q"]]
        if not checked_empty:                                   # Submit with nothing chosen: stays on the same question and warns
            click("Submit answer"); assert at.session_state["i"] == i and any("choose or type" in w.value for w in at.warning), [w.value for w in at.warning]; checked_empty = True
        if q["type"] == "short": at.text_input[0].set_value("a lock prevents two threads racing"); at.run(); click("Submit answer"); n_sub += 1
        elif i % 3 == 0: at.session_state["t_step"] = time.time() - 999; at.run(); n_timeout += 1             # time-out
        elif i % 3 == 1: at.radio[0].set_value(at.radio[0].options[0]); at.run(); assert at.session_state["i"] == i, "selecting must not advance"; click("Submit answer"); n_sub += 1
        else: click("Skip"); n_skip += 1
        assert at.session_state["i"] == i + 1, (k, i)
    elif k == "post": click("Submit")
    else: at.session_state["t_step"] = time.time() - 999; at.run()
print("step kinds exercised:", sorted(seen), f"| submitted {n_sub}, skipped {n_skip}, timed out {n_timeout} | final stage:", at.session_state["stage"])
assert not at.exception, at.exception
assert not at.session_state["jobs"] and len(at.session_state["pending"]) == 10, "free grader must run after the interview, not during it"
assert at.session_state["rep"]["technical"]["overall"] >= 0 and os.path.exists(f"data/sessions/{at.session_state['sid']}.json")
print("done page + saved session OK:", at.session_state["sid"])
os.remove(f"data/sessions/{at.session_state['sid']}.json"); os.remove("tests/_app_fake.py"); os.remove("data/bank.json")
