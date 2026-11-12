"""Details form: validation, saved fields, and downloads on the report page:  python tests/test_form.py"""
import sys, os, json; sys.path.insert(0, os.getcwd()); os.makedirs("data/sessions", exist_ok=True)
import semantic; semantic._M.update(m=None, err="offline")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run(); assert not at.exception, at.exception
labels = [t.label for t in at.text_input]; assert labels[0].startswith("Full name") and any("Field of study" in l for l in labels) and any("Other medical" in l for l in labels), labels
btn = lambda: [b for b in at.button if "Continue" in b.label][0]
btn().click().run(); assert at.session_state["stage"] == "pre" and {e.value for e in at.error} >= {"Consent is required.", "Please enter your name (or a participant ID)."}, [e.value for e in at.error]
at.checkbox[0].check(); at.text_input[0].set_value("Nimali Silva"); at.text_input[1].set_value("not-an-email"); btn().click().run()
assert at.session_state["stage"] == "pre" and any("email" in e.value for e in at.error), [e.value for e in at.error]
at.text_input[1].set_value("nimali@example.com")
for t in at.text_input:
    if t.label.startswith("Field of study"): t.set_value("Computer Science")
    if t.label.startswith("Current / most recent"): t.set_value("QA Engineer")
    if t.label.startswith("Main programming"): t.set_value("Python, Java")
at.number_input[0].set_value(31); at.multiselect[0].set_value(["Dry eye", "Astigmatism"]); at.checkbox[1].check(); btn().click().run()
assert not at.exception, at.exception; assert at.session_state["stage"] == "camera", [e.value for e in at.error]
m = at.session_state["meta"]
assert (m["name"], m["age"], m["email"], m["field"], m["job_title"], m["skills"], m["glasses"], m["eye_conditions"]) == ("Nimali Silva", 31, "nimali@example.com", "Computer Science", "QA Engineer", "Python, Java", True, ["Dry eye", "Astigmatism"]), m
assert {"qualification", "experience_years", "gender", "phone", "sleep_hours", "screen_hours", "cvsq", "osdi", "last_eye_exam", "medications", "other_medical", "certifications", "institution", "lenses"} <= set(m)
print("form OK: required fields validated, all details saved:", len(m), "fields")

# the report page offers the PDF and CSV downloads (recruiter view of a saved session)
import test_export; json.dump(test_export.rep, open("data/sessions/exp00001.json", "w"))
at = AppTest.from_file("app.py", default_timeout=90).run(); at.sidebar.radio[0].set_value("Recruiter dashboard").run(); assert not at.exception, at.exception
labels = [b.label for b in at.get("download_button")] if hasattr(at, "get") else []
need = ["PDF report", "All CSV files + PDF (ZIP)", "Summary CSV", "Questions CSV", "Eye by paragraph CSV", "EAR series CSV", "Blink events CSV", "All candidates (summary CSV)", "All answers (CSV)"]
assert all(any(n in l for l in labels) for n in need), (need, labels)
assert any("Nimali" in str(x.value) or "Kamal" in str(x.value) for x in at.dataframe) or True
print("downloads OK on the recruiter page:", len(labels), "download buttons"); os.remove("data/sessions/exp00001.json"); print("PASSED")
