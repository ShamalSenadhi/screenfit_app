"""Model page + JSON extraction (no real model needed):  python tests/test_model_page.py"""
import sys, os; sys.path.insert(0, os.getcwd()); os.makedirs("data", exist_ok=True)
import llm
assert llm.extract_json('Sure! ```json\n{"a": 1, "b": {"c": "}"}}\n``` hope that helps') == {"a": 1, "b": {"c": "}"}}
assert llm.extract_json('{"correctness": 2, "completeness": 1, "justification": "ok"} trailing') ["correctness"] == 2
try: llm.extract_json("no json here"); raise SystemExit("should fail")
except ValueError: pass
s = llm.status("meta-llama/Llama-3.2-3B-Instruct"); assert set(s) >= {"libs", "cuda", "token", "cached", "loaded"}; print("status:", s)
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run(); at.sidebar.radio[0].set_value("Model setup (Llama)").run()
assert not at.exception, at.exception
assert any("Libraries" in m.label for m in at.metric) and any("downloads" in c.value for c in at.caption)
calls = []
llm.ask = lambda system, user, temperature=0.0, model=None, max_new=700: (calls.append(model) or {"ok": True, "reply": "hello"})
at.run(); [b for b in at.button if "JSON test" in b.label][0].click().run()          # disabled when torch is missing
at.sidebar.radio[0].set_value("Item bank (admin)").run(); assert not at.exception, at.exception
print("Model setup page + admin page OK, no download button:", not any("ownload" in b.label for b in at.button)); print("PASSED")
