"""Exercises every interview step with a fake camera (no webcam):  python tests/test_run_stage.py"""
import sys, os, re, time, types
sys.path.insert(0, os.getcwd()); os.makedirs("data/sessions", exist_ok=True)
import numpy as np, pandas as pd, random
import llm, bank
llm.load = lambda m=None: None; llm.is_loaded = lambda m=None: True      # no real model in tests
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
steps = at.session_state["steps"]; assert sum(s["dur"] for s in steps) == 1200, sum(s["dur"] for s in steps)
print("steps:", len(steps), "total", sum(s["dur"] for s in steps), "s; question types per paragraph:", [s["dur"] for s in steps if s["k"] == "q"][:5])
# walk every step by answering via widgets (early advance) and by forcing time-outs
seen = set(); guard = 0
while at.session_state["stage"] == "run" and guard < 60:
    guard += 1; i = at.session_state["i"]; k = steps[i]["k"]; seen.add(k); assert not at.exception, (k, at.exception)
    if k == "q":
        q = at.session_state["qs"][steps[i]["p"]]["questions"][steps[i]["q"]]
        if q["type"] == "short": at.text_input[0].set_value("a lock prevents two threads racing"); at.run()
        elif i % 2: at.radio[0].set_value(at.radio[0].options[0]).run()
        else: at.session_state["t_step"] = time.time() - 999; at.run()      # force a time-out
    elif k == "post": [bt for bt in at.button if "Submit" in bt.label][0].click().run()
    else: at.session_state["t_step"] = time.time() - 999; at.run()
print("step kinds exercised:", sorted(seen), "| final stage:", at.session_state["stage"])
assert not at.exception, at.exception
assert at.session_state["rep"]["technical"]["overall"] >= 0 and os.path.exists(f"data/sessions/{at.session_state['sid']}.json")
print("done page + saved session OK:", at.session_state["sid"])
os.remove(f"data/sessions/{at.session_state['sid']}.json"); os.remove("tests/_app_fake.py"); os.remove("data/bank.json")
