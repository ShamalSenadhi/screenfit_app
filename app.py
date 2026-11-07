"""ScreenFit - LLM interview (Llama via Hugging Face) + webcam eye screening. Run:  streamlit run app.py"""
import os, glob, json, time, random, uuid
import numpy as np, pandas as pd, streamlit as st
from concurrent.futures import ThreadPoolExecutor
from streamlit_autorefresh import st_autorefresh
import bank, llm, features, report

st.set_page_config(page_title="ScreenFit", page_icon="👁️", layout="wide")
S = st.session_state
RTC = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
GRID = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
TOTAL = 1200; READ_S = 35; QDUR = {"mcq": 20, "tf": 20, "short": 35}
NEUTRAL = ("Please read this neutral text silently at your normal pace. Rivers carry water from high ground to the sea. "
           "Along the way they shape valleys, feed forests and provide a path for travellers. Seasons change the flow: spring brings "
           "fast water, summer slows it, and autumn leaves gather along the banks. Keep your head still and your eyes on the screen.")
os.makedirs("data/sessions", exist_ok=True)
S.setdefault("stage", "pre"); S.setdefault("model", llm.MODEL)


def build_steps(items):
    s = [dict(k="setup", dur=60), dict(k="calib", dur=90), dict(k="baseline", dur=60)]
    for p in range(5):
        s.append(dict(k="read", p=p, dur=READ_S))
        s += [dict(k="q", p=p, q=q, dur=QDUR[items[p]["questions"][q]["type"]]) for q in range(5)]
    return s + [dict(k="post", dur=105), dict(k="wrap", dur=60)]          # sums to 1200 s

def advance(timeout=False):
    stp = S.steps[S.i]
    if stp["k"] == "q":
        q = S.qs[stp["p"]]["questions"][stp["q"]]; raw = S.get(f"a_{S.i}"); key = f"{stp['p']}-{stp['q']}"; val = None
        if raw not in (None, ""):
            if q["type"] == "mcq": val = int("ABCD".index(raw[0]))
            elif q["type"] == "tf": val = (raw == "True")
            else: val = raw.strip()
        S.ans[key] = dict(val=val, timed_out=bool(timeout and val is None))
        if q["type"] == "short" and val: S.jobs[key] = S.pool.submit(llm.grade_short, q, val, 3, S.model)
    S.i += 1; S.t_step = time.time()

def cam():
    from streamlit_webrtc import webrtc_streamer
    import eyes
    with st.sidebar:
        st.markdown("### Webcam (processed locally)")
        ctx = webrtc_streamer(key="eye", video_processor_factory=eyes.EyeProcessor, rtc_configuration=RTC,
                              media_stream_constraints={"video": {"width": 480, "height": 360}, "audio": False})
    return ctx, ctx.video_processor

def finalize(proc):
    feats = None
    try: feats = features.compute(proc.df()) if proc is not None else None
    except Exception as e: st.warning(f"Eye analysis failed: {e}")
    grades = {}
    for k, f in S.jobs.items():
        try: grades[k] = f.result(timeout=120)
        except Exception: grades[k] = None
    rep = report.build(S.sid, S.seed, S.meta, S.qs, S.ans, grades, feats, S.get("post", {}))
    json.dump(rep, open(f"data/sessions/{S.sid}.json", "w"), indent=1); S.rep = rep; S.stage = "done"

# ------------------------------------------------------------------ candidate
def candidate():
    st.title("👁️ ScreenFit · 20-minute technical interview")
    if S.stage == "pre":
        st.subheader("Pre-session form (not counted in the 20 minutes)")
        with st.form("pre_form"):
            ok = st.checkbox("I have read the information sheet and agree to take part. My webcam is analysed on this device; no video is stored; this is not a medical diagnosis; I may withdraw at any time.")
            c1, c2, c3 = st.columns(3)
            level = c1.selectbox("Role level", ["Junior", "Mid-level", "Senior"]); glasses = c2.checkbox("I wear glasses"); hrs = c3.number_input("Screen hours per day", 0.0, 16.0, 8.0, 0.5)
            st.markdown("Complete the official **CVS-Q** and **OSDI** questionnaires and enter your total scores:")
            c1, c2 = st.columns(2); cvsq = c1.number_input("CVS-Q total score", 0, 100, 0); osdi = c2.number_input("OSDI total score (0-100)", 0.0, 100.0, 0.0, 1.0)
            go = st.form_submit_button("Continue to camera check", type="primary")
        if go:
            b = bank.load()
            try: seed = random.randrange(2**31); items = bank.assemble(b, seed)
            except Exception as e: st.error(f"Item bank not ready: {e}. Use the 'Item bank (admin)' page first."); return
            if not ok: st.error("Consent is required."); return
            S.update(meta=dict(level=level, glasses=glasses, screen_hours=hrs, cvsq=cvsq, osdi=osdi), qs=items, seed=seed, sid=uuid.uuid4().hex[:8], stage="camera"); st.rerun()
    elif S.stage == "camera":
        if not llm.is_loaded(S.model):
            with st.spinner("Loading Llama for grading (first use downloads it automatically; please wait)…"):
                try: llm.load(S.model)
                except Exception as e: st.error(f"Could not load the model: {e}. Open **Model setup (Llama)** (token, licence, GPU memory)."); return
        ctx, proc = cam(); st_autorefresh(1500, key="camtick")
        st.subheader("Camera check"); st.write("Click **START** in the sidebar, allow the camera, face the screen with front lighting, and sit about 60 cm away.")
        ok = False
        if proc is not None:
            d = proc.df().tail(45); ok = len(d) > 10 and d.ok.mean() > 0.9
            st.write(("✅ Face detected" if ok else "⚠️ Face not detected reliably") + (f" ({d.ok.mean()*100:.0f}% of recent frames)" if len(d) else ""))
        if st.button("Begin the 20-minute interview", type="primary", disabled=not (ctx.state.playing and ok)):
            S.update(steps=build_steps(S.qs), i=0, ans={}, jobs={}, pool=ThreadPoolExecutor(2), t_start=time.time(), t_step=time.time(), stage="run"); st.rerun()
    elif S.stage == "run":
        ctx, proc = cam(); st_autorefresh(1000, key="tick"); run(proc)
    else:
        st.success("Interview complete. Thank you!"); report.render(S.rep, candidate=True)
        if st.button("Start a new session"):
            for k in ["qs", "steps", "ans", "jobs", "rep", "post", "meta"]: S.pop(k, None)
            S.stage = "pre"; st.rerun()

def run(proc):
    now = time.time(); stp = S.steps[S.i]; el = now - S.t_step; rem = stp["dur"] - el
    if rem <= 0 and stp["k"] not in ("wrap",):
        advance(timeout=True); st.rerun()
    if proc is None: st.warning("Waiting for the webcam stream…"); return
    st.progress(min(1.0, (now - S.t_start) / TOTAL), text=f"Interview time left: {int(max(0, TOTAL - (now - S.t_start)) // 60):02d}:{int(max(0, TOTAL - (now - S.t_start)) % 60):02d}  ·  step {S.i + 1}/{len(S.steps)}  ·  this step: {max(0, int(rem))} s")
    k = stp["k"]
    if k == "setup":
        proc.label = "setup"; d = proc.df().tail(60); st.subheader("1 · Set-up check")
        st.write(f"Face detected: **{d.ok.mean()*100:.0f}%**" if len(d) else "No frames yet"); st.write("Keep your head centred, light in front of you, no strong light behind you.")
    elif k == "calib":
        sub = min(8, int(el // 10)); proc.label = "calib"; proc.target = GRID[sub]
        st.subheader(f"2 · Gaze calibration — look at the dot ({sub + 1}/9)")
        cells = "".join(f"<div style='display:flex;align-items:center;justify-content:center;height:90px'><div style='width:{36 if n == sub else 12}px;height:{36 if n == sub else 12}px;border-radius:50%;background:{'#d64545' if n == sub else '#c5cde0'}'></div></div>" for n in range(9))
        st.markdown(f"<div style='display:grid;grid-template-columns:repeat(3,1fr);border:1px dashed #ccd;border-radius:12px;padding:20px'>{cells}</div>", unsafe_allow_html=True)
    elif k == "baseline":
        proc.label = "baseline"; proc.target = (np.nan, np.nan); st.subheader("3 · Baseline reading"); st.markdown(f"<div style='font:20px/2 Georgia,serif;max-width:70ch'>{NEUTRAL}</div>", unsafe_allow_html=True)
    elif k == "read":
        p = stp["p"]; it = S.qs[p]; proc.label = f"read{p + 1}"; proc.target = (np.nan, np.nan)
        st.subheader(f"Paragraph {p + 1} of 5 — read carefully ({READ_S} s)")
        st.markdown(f"<div style='font:20px/2 Georgia,serif;max-width:70ch'>{it['paragraph']}</div>", unsafe_allow_html=True)
    elif k == "q":
        p, qi = stp["p"], stp["q"]; q = S.qs[p]["questions"][qi]; proc.label = f"q{p + 1}"; key = f"a_{S.i}"
        st.subheader(f"Paragraph {p + 1} · Question {qi + 1} of 5"); st.write(f"**{q['text']}**")
        if q["type"] == "mcq": st.radio("Choose one", [f"{'ABCD'[n]}. {o}" for n, o in enumerate(q["options"])], index=None, key=key, on_change=advance)
        elif q["type"] == "tf": st.radio("True or false?", ["True", "False"], index=None, horizontal=True, key=key, on_change=advance)
        else: st.text_input("Answer in at most 20 words, then press Enter", max_chars=160, key=key, on_change=advance)
        st.caption("The paragraph is no longer shown. Answers are final once submitted.")
    elif k == "post":
        proc.label = "post"; st.subheader("Post-task questions")
        with st.form("post_form"):
            tl = [st.slider(n, 0, 100, 50) for n in ["Mental demand", "Physical demand", "Temporal demand", "Performance (100 = poor)", "Effort", "Frustration"]]
            strain = st.slider("Eye strain right now (0 = none, 10 = severe)", 0, 10, 2)
            if st.form_submit_button("Submit", type="primary"): S.post = dict(tlx=round(float(np.mean(tl)), 1), strain=strain); advance(); st.rerun()
    else:  # wrap
        proc.label = "post"
        with st.spinner("Scoring answers (Llama) and analysing eye signals…"): finalize(proc)
        st.rerun()


# ------------------------------------------------------------------ model setup / diagnostics
PRESETS = ["meta-llama/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]

def setup():
    st.title("🛠️ Model setup · Llama (Hugging Face)")
    st.caption("The model downloads **automatically** the first time it is loaded (progress appears in the terminal where Streamlit runs). There is no download button.")
    labels = ["Llama-3.2-11B-Vision-Instruct (vision; GPU, ~10 GB VRAM in 4-bit)", "Llama-3.2-3B-Instruct (text; light)", "Llama-3.2-1B-Instruct (text; testing only)", "other…"]
    pick = st.selectbox("Model", labels, index=PRESETS.index(S.model) if S.model in PRESETS else 3)
    S.model = ("meta-llama/" + pick.split(" ")[0]) if not pick.startswith("other") else (st.text_input("Hugging Face model id", S.model).strip() or llm.MODEL)
    tok = st.text_input("Hugging Face access token (Llama is gated: accept the licence on the model page first)", value=os.getenv("HF_TOKEN", ""), type="password")
    if tok: os.environ["HF_TOKEN"] = tok.strip()
    s = llm.status(S.model); c = st.columns(5)
    c[0].metric("Libraries", "✅" if s["libs"] else "❌"); c[1].metric("GPU", f"✅ {s['vram_gb']} GB" if s["cuda"] else "none (CPU)")
    c[2].metric("HF token", "✅" if s["token"] else "❌"); c[3].metric("Cached on disk", "✅" if s["cached"] else "not yet"); c[4].metric("Loaded", "✅" if s["loaded"] else "no")
    if not s["libs"]: st.error("Install the libraries:  pip install torch transformers accelerate bitsandbytes huggingface_hub  (then restart the app)")
    if "vision" in S.model.lower() and not s["cuda"]: st.warning("Llama 3.2 Vision (11B) is far too slow on CPU. Use a CUDA GPU, or choose Llama-3.2-3B-Instruct.")
    st.info("Llama models are gated: open the model page on huggingface.co, accept the licence, create a read token and paste it above (or set HF_TOKEN). The first load downloads the weights once; later runs use the cache.")
    c1, c2 = st.columns(2)
    if c1.button("Load model and run JSON test", disabled=not s["libs"]):
        with st.spinner("Loading the model (first use downloads it automatically; this can take a long time)…"):
            try: r = llm.ask("Reply with JSON only.", 'Return {"ok": true, "reply": "hello"}', 0, S.model, max_new=40); S["_test"] = "ok"; st.success("Model works."); st.json(r)
            except Exception as e: st.error(f"Failed: {e}")
    if c2.button("Generate one sample item", disabled=not s["libs"]):
        with st.spinner("Generating…"): it = bank.make_item("REST API design", "Data and APIs", S.model)
        if it: st.success(f"Valid item generated ({it['words']} words)"); st.write(it["paragraph"]); st.json(it["questions"])
        else: st.error("No valid item after 3 tries. Try a larger model or generate again.")

# ------------------------------------------------------------------ admin
def admin():
    st.title("🗂️ Item bank (admin)")
    s = llm.status(S.model)
    st.info(f"Model: `{S.model}` · cached: {'yes' if s['cached'] else 'no (downloads automatically on first use)'} · loaded: {'yes' if s['loaded'] else 'no'}. Change it in **Model setup (Llama)**.")
    ms = [S.model] if s["libs"] else []
    if not s["libs"]: st.error("PyTorch/transformers are not installed. See **Model setup (Llama)**.")
    b = bank.load(); cov = bank.coverage(b); st.write("**Approved items per cluster**"); st.bar_chart(pd.Series(cov, name="approved"))
    with st.expander("Generate new items with Llama", expanded=not b):
        cl = st.multiselect("Clusters", list(bank.CLUSTERS), default=list(bank.CLUSTERS)); n = st.slider("Items per area", 1, 5, 1)
        if st.button("Generate", type="primary", disabled=not ms):
            areas = [(a, c) for c in cl for a in bank.CLUSTERS[c]] * n; bar = st.progress(0.0); made = 0
            for j, (a, c) in enumerate(areas):
                it = bank.make_item(a, c, S.model)
                if it: b.append(it); made += 1
                bar.progress((j + 1) / len(areas), text=f"{a} ({j + 1}/{len(areas)})")
            bank.save(b); st.success(f"Generated {made} of {len(areas)} items. Review and approve them below."); st.rerun()
    if b:
        df = pd.DataFrame([{k: i[k] for k in ["id", "cluster", "area", "words", "fk", "approved"]} for i in b])
        ed = st.data_editor(df, hide_index=True, use_container_width=True, disabled=["id", "cluster", "area", "words", "fk"], key="bank_ed")
        c1, c2 = st.columns(2)
        if c1.button("Save approvals"):
            ap = dict(zip(ed.id, ed.approved)); [i.update(approved=bool(ap[i["id"]])) for i in b]; bank.save(b); st.success("Saved."); st.rerun()
        if c2.button("Delete unapproved items"): bank.save([i for i in b if i["approved"]]); st.rerun()
        pick = st.selectbox("Preview item", [i["id"] for i in b]); it = next(i for i in b if i["id"] == pick)
        st.write(it["paragraph"]); [st.write(f"**{q['type'].upper()}** {q['text']}  \n" + (f"Options: {q['options']} → {'ABCD'[q['answer']]}" if q["type"] == "mcq" else f"Answer: {q.get('answer', q.get('key'))}")) for q in it["questions"]]

# ------------------------------------------------------------------ recruiter
def recruiter():
    st.title("📊 Recruiter dashboard")
    reps = [json.load(open(f)) for f in sorted(glob.glob("data/sessions/*.json"))]
    if not reps: st.info("No completed sessions yet."); return
    tab = pd.DataFrame([dict(session=r["sid"], when=r["when"], level=r["meta"].get("level"), score=round(r["technical"]["overall"]), risk=r["visual"]["risk"],
                             face_ok=(r["visual"].get("quality") or {}).get("face_ok")) for r in reps])
    c = st.columns(4); c[0].metric("Sessions", len(tab)); c[1].metric("Mean technical score", f"{tab.score.mean():.0f}")
    c[2].metric("Moderate/high visual risk", int((tab.risk != "Low").sum())); c[3].metric("Mean face detected", f"{tab.face_ok.mean():.0f}%" if tab.face_ok.notna().any() else "n/a")
    a, b = st.columns(2); a.subheader("Score distribution"); a.bar_chart(tab.score.value_counts(bins=5).sort_index().rename(lambda x: str(x)))
    b.subheader("Visual-risk levels (shown for context; never used to rank)"); b.bar_chart(tab.risk.value_counts())
    st.dataframe(tab, hide_index=True, use_container_width=True)
    sel = st.selectbox("Open candidate session", tab.session); st.divider(); report.render(next(r for r in reps if r["sid"] == sel))

mode = st.sidebar.radio("Mode", ["Candidate interview", "Item bank (admin)", "Recruiter dashboard", "Model setup (Llama)"])
if S.stage != "run": st.sidebar.caption(f"Llama: {S.model.split('/')[-1]} · " + ("✅ loaded" if llm.is_loaded(S.model) else "loads on first use"))
if mode != "Candidate interview" and S.stage in ("camera", "run"): st.sidebar.warning("An interview is in progress.")
{"Candidate interview": candidate, "Item bank (admin)": admin, "Recruiter dashboard": recruiter, "Model setup (Llama)": setup}[mode]()
