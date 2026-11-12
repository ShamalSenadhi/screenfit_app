"""PDF report + CSV exports:  python tests/test_export.py   (set PDF_OUT=/path/report.pdf to keep the PDF for viewing)"""
import sys, os, io, re, json, zipfile; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, bank, features, report, export, seed_bank
rng = np.random.default_rng(2); rows = []; t = 100.0
def seg(label, secs, every, tgt=None, scan=False):
    global t
    for n in range(int(secs * 25)):
        e = 0.30 + 0.006 * rng.standard_normal(); e = 0.07 if (n / 25) % every < 0.15 else e
        sx, sy = ((n / 25 * 0.35) % 1.0, 0.3 + 0.1 * int(n / 25 / 3)) if scan else (tgt or (0.5, 0.5))
        rows.append(dict(t=t, label=label, tx=tgt[0] if tgt else np.nan, ty=tgt[1] if tgt else np.nan, ok=1, ear=e, ix=.3 + .4 * sx + .004 * rng.standard_normal(), iy=.3 + .4 * sy, yaw=.1 * sx, pitch=.2 * sy)); t += 1 / 25
seg("setup", 5, 4)
for tg in [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]: seg("calib", 7, 4, tgt=tg)
seg("baseline", 45, 4)
for p in range(1, 6): seg(f"read{p}", 35, 4 + p, scan=True); seg(f"q{p}", 60, 4 + p)
seg("post", 20, 5)
feats = features.compute(pd.DataFrame(rows)); items = bank.assemble(seed_bank.items(), 11); ans, grades = {}, {}
for pi, it in enumerate(items):
    for qi, q in enumerate(it["questions"]):
        k = f"{pi}-{qi}"
        if q["type"] == "short": ans[k] = dict(val="a lock prevents two threads racing on shared data", timed_out=False); grades[k] = dict(score=0.75, why="Mentions the key idea.", agree=True)
        else: ans[k] = dict(val=q["answer"] if qi != 1 else None, timed_out=(qi == 1))
meta = dict(name="Kamal Perera \u0dc1\u0dcf\u0db8\u0dca", age=27, gender="Male", email="kamal@example.com", phone="+94 77 000 0000", qualification="Bachelor's degree", field="Software Engineering", institution="University of Moratuwa",
            experience_years=3.5, job_title="Software Engineer", level="Mid-level", skills="Python, SQL, React", certifications="AWS Cloud Practitioner", glasses=True, lenses=False,
            eye_conditions=["Short-sightedness (myopia)", "Dry eye"], last_eye_exam="Within 1 year", other_medical="None", medications="", sleep_hours=6.5, screen_hours=9.0, cvsq=7, osdi=19, grader="free semantic grader", capture="browser")
rep = report.build("exp00001", 11, meta, items, ans, grades, feats, dict(tlx=47.5, strain=4)); rep = json.loads(json.dumps(rep))

pdf = export.pdf_bytes(rep); assert pdf[:5] == b"%PDF-" and len(pdf) > 8000
pages = len(re.findall(rb"/Type\s*/Page\b", pdf)); assert 3 <= pages <= 8, pages
if os.environ.get("PDF_OUT"): open(os.environ["PDF_OUT"], "wb").write(pdf)
txt = os.popen("pdftotext -layout - -", "w") if False else None
import subprocess; out = subprocess.run(["pdftotext", "-layout", "-", "-"], input=pdf, capture_output=True).stdout.decode("utf-8", "replace") if subprocess.run(["which", "pdftotext"], capture_output=True).returncode == 0 else ""
for needle in ("Kamal Perera", "University of Moratuwa", "Moderate risk", "Technical score", "EAR and frame data", "Baseline EAR", "screening indicator", "Questions and answers"):
    assert not out or needle in out, needle
print(f"PDF OK: {pages} pages, {len(pdf):,} bytes, non-Latin name handled without crashing")

s = export.summary_row(rep)
for k in ("session", "name", "age", "qualification", "experience_years", "eye_conditions", "technical_score", "visual_risk", "frames_total", "baseline_ear", "blinks_total", "blink_rate_per_min"): assert k in s and s[k] is not None, k
assert s["eye_conditions"] == "Short-sightedness (myopia), Dry eye" and s["name"].startswith("Kamal") and s["timed_out_questions"] == 5
assert any(c.startswith("score_") for c in s) and 40 <= s["frames_total"] / 100 and s["blinks_total"] > 50
z = zipfile.ZipFile(io.BytesIO(export.zip_bytes(rep, pdf))); names = sorted(z.namelist())
assert names == ["blink_events.csv", "ear_series.csv", "eye_by_paragraph.csv", "questions.csv", "report.pdf", "summary.csv"], names
q = pd.read_csv(io.BytesIO(z.read("questions.csv")), encoding="utf-8-sig"); assert len(q) == 25 and q.name.iloc[0].startswith("Kamal") and "\u0dc1" in q.name.iloc[0]
assert len(pd.read_csv(io.BytesIO(z.read("ear_series.csv")))) > 500 and len(pd.read_csv(io.BytesIO(z.read("eye_by_paragraph.csv")))) == 5
two = export.summary_df([rep, dict(rep, sid="exp00002")]); assert len(two) == 2 and list(two.session) == ["exp00001", "exp00002"]
old = json.loads(json.dumps(rep)); old["meta"] = dict(level="Junior", glasses=False, screen_hours=8.0, cvsq=0, osdi=0); old["visual"]["ear"] = None; old["visual"]["para"] = []
assert export.pdf_bytes(old)[:5] == b"%PDF-" and len(export.zip_bytes(old)) > 500          # older sessions without profile / eye data still export
print(f"CSV OK: summary row with {len(s)} columns, zip with {len(names)} files, Unicode kept, older sessions export"); print("PASSED")
