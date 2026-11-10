"""ScreenFit - LLM interview (Llama via hosted API) + webcam eye screening. Run:  streamlit run app.py"""
import os, glob, json, time, random, uuid
import numpy as np, pandas as pd, streamlit as st
from concurrent.futures import ThreadPoolExecutor
from streamlit_autorefresh import st_autorefresh
import bank, llm, features, report, grader, semantic

st.set_page_config(page_title="ScreenFit", page_icon="👁️", layout="wide")
for _m, _n in ((bank, "make_item"), (llm, "ask"), (features, "compute"), (report, "build"), (grader, "grade"), (semantic, "warm")):
    if not hasattr(_m, _n): st.error(f"`{_m.__name__}.py` has the wrong content (missing `{_n}`). Re-copy it from the project zip."); st.stop()
S = st.session_state
def get_rtc():
    """STUN by default. On Streamlit Cloud add TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN secrets for a TURN server (more reliable)."""
    c = S.get("_rtc")
    if c and time.time() - c[0] < 3000: return c[1]
    cfg = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]},      # public relay fallback (best effort); set TURN_URL/TURN_USERNAME/TURN_CREDENTIAL or Twilio for a reliable one
                          {"urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443", "turns:openrelay.metered.ca:443?transport=tcp"], "username": "openrelayproject", "credential": "openrelayproject"}]}
    if os.getenv("TURN_URL"): cfg["iceServers"].append({"urls": [os.getenv("TURN_URL")], "username": os.getenv("TURN_USERNAME", ""), "credential": os.getenv("TURN_CREDENTIAL", "")})
    sid, tok = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if sid and tok:
        try:
            import requests
            r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Tokens.json", auth=(sid, tok), timeout=10); r.raise_for_status()
            cfg = {"iceServers": r.json()["ice_servers"]}
        except Exception: pass
    S["_rtc"] = (time.time(), cfg); return cfg
GRID = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
SETUP_S, CAL_DOT_S, BASE_S = 45, 8, 45          # shorter initial phases (was 60 / 90 / 60); calibration = 9 dots x 8 s = 72 s
CALIB_S = 9 * CAL_DOT_S; READ_S = 35; QDUR = {"mcq": 20, "tf": 20, "short": 35}; POST_S, WRAP_S = 105, 60
TOTAL = SETUP_S + CALIB_S + BASE_S + 5 * (READ_S + 2 * QDUR["mcq"] + QDUR["tf"] + 2 * QDUR["short"]) + POST_S + WRAP_S      # 1152 s
NEUTRAL = ("Please read this neutral text silently at your normal pace. Rivers carry water from high ground to the sea. "
           "Along the way they shape valleys, feed forests and provide a path for travellers. Seasons change the flow: spring brings "
           "fast water, summer slows it, and autumn leaves gather along the banks. Keep your head still and your eyes on the screen.")
os.makedirs("data/sessions", exist_ok=True)
S.setdefault("stage", "pre"); S.setdefault("model", llm.MODEL)


def build_steps(items):
    s = [dict(k="setup", dur=SETUP_S), dict(k="calib", dur=CALIB_S), dict(k="baseline", dur=BASE_S)]
    for p in range(5):
        s.append(dict(k="read", p=p, dur=READ_S))
        s += [dict(k="q", p=p, q=q, dur=QDUR[items[p]["questions"][q]["type"]]) for q in range(5)]
    return s + [dict(k="post", dur=POST_S), dict(k="wrap", dur=WRAP_S)]          # sums to TOTAL

def advance(timeout=False):
    stp = S.steps[S.i]
    if stp["k"] == "q":
        q = S.qs[stp["p"]]["questions"][stp["q"]]; raw = S.get(f"a_{S.i}"); key = f"{stp['p']}-{stp['q']}"; val = None
        if raw not in (None, ""):
            if q["type"] == "mcq": val = int("ABCD".index(raw[0]))
            elif q["type"] == "tf": val = (raw == "True")
            else: val = raw.strip()
        S.ans[key] = dict(val=val, timed_out=bool(timeout and val is None))
        if q["type"] == "short" and val: S.pending[key] = (q, val)        # every short answer is graded only after the interview is over (camera keeps the CPU and memory)
    S.i += 1; S.t_step = time.time()

def cam():
    """Camera source. Default: eye capture inside the browser (no WebRTC server, no TURN, video never leaves the device)."""
    if S.get("capture", "browser") == "browser":
        import types, eyecomp
        if not isinstance(S.get("eye"), eyecomp.BrowserEyes): S.eye = eyecomp.BrowserEyes()
        with st.sidebar:
            st.markdown("### Camera (eye numbers only, nothing is recorded)")
            eyecomp.render(S.eye)
            if S.eye.error: st.error(f"Camera/eye model problem: {S.eye.error}. Allow the camera, use Chrome or Edge, or switch to *Server (WebRTC)* on the camera page.")
        return types.SimpleNamespace(state=types.SimpleNamespace(playing=S.eye.playing)), S.eye
    try:
        from streamlit_webrtc import webrtc_streamer
        import eyes
    except Exception as e:
        st.error(f"The eye-measurement library could not be loaded: `{type(e).__name__}: {e}`. On Streamlit Cloud add a file `packages.txt` (next to requirements.txt) with the two lines `libgl1` and `libglib2.0-0`, make sure requirements.txt has no other opencv package, then reboot the app.")
        st.stop()
    with st.sidebar:
        st.markdown("### Webcam (live preview only, nothing is recorded)")
        ctx = webrtc_streamer(key="eye", video_processor_factory=eyes.EyeProcessor, rtc_configuration=get_rtc(),
                              media_stream_constraints={"video": {"width": {"ideal": 320}, "height": {"ideal": 240}, "frameRate": {"ideal": 20, "max": 24}}, "audio": False},
                              async_processing=True)
    return ctx, ctx.video_processor

def finalize(proc):
    feats = None
    try: feats = features.compute(proc.df()) if proc is not None else None
    except Exception as e: st.warning(f"Eye analysis failed: {e}")
    if proc is not None and hasattr(proc, "release"): proc.release()          # free MediaPipe before any heavy grading model loads
    grades = {}
    for k, f in S.jobs.items():
        try: grades[k] = f.result(timeout=120)
        except Exception: grades[k] = None
    pend = S.get("pending", {})
    if pend:
        bar = st.progress(0.0, text="Grading short answers…")
        for n, (k, (q, val)) in enumerate(pend.items()):
            try: grades[k] = grader.grade(q, val, S.get("grader", "semantic"), S.model)
            except Exception: grades[k] = None
            bar.progress((n + 1) / len(pend), text=f"Grading short answers {n + 1}/{len(pend)}")
    S.meta["grader"] = semantic.engine() if S.get("grader", "semantic") == "semantic" else f"LLM {S.model}"
    rep = report.build(S.sid, S.seed, S.meta, S.qs, S.ans, grades, feats, S.get("post", {}))
    json.dump(rep, open(f"data/sessions/{S.sid}.json", "w"), indent=1); S.rep = rep; S.stage = "done"

# ------------------------------------------------------------------ candidate
def candidate():
    st.title("👁️ ScreenFit · 19-minute technical interview")
    if S.stage == "pre":
        st.subheader("Pre-session form (not counted in the interview time)")
        with st.form("pre_form"):
            ok = st.checkbox("I have read the information sheet and agree to take part. My webcam is analysed inside my browser (default) or, if I choose server capture, in memory on the server; video is never recorded or stored (only eye-measurement numbers such as EAR and blink counts are kept); this is not a medical diagnosis; I may withdraw at any time.")
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
        if S.get("grader", "semantic") == "llm" and not llm.ready(): st.error("The selected LLM needs an API key. Open **Model setup** (or switch to the free grader)."); return
        st.caption("Short-answer grader: " + ("free semantic grader (loaded after the interview)" if S.get("grader", "semantic") == "semantic" else f"LLM · {llm.CFG['provider']}"))
        S.capture = st.radio("Camera capture", ["browser", "server"], index=0 if S.get("capture", "browser") == "browser" else 1, horizontal=True,
                             format_func=lambda x: {"browser": "Browser (recommended: video stays on your device)", "server": "Server (WebRTC, needs a TURN server online)"}[x])
        ctx, proc = cam(); st_autorefresh(1500, key="camtick")
        st.subheader("Camera check"); st.write("Click **START** in the sidebar, allow the camera, face the screen with front lighting, and sit about 60 cm away.")
        ok = False
        if proc is not None:
            d = proc.df().tail(45); ok = len(d) > 10 and d.ok.mean() > 0.9
            st.write(("✅ Face detected" if ok else "⚠️ Face not detected reliably") + (f" ({d.ok.mean()*100:.0f}% of recent frames)" if len(d) else ""))
            st.caption(f"Camera diagnostics: frames received {getattr(proc, 'received', 0)} · frame errors {getattr(proc, 'errors', 0)} (if 'received' stays at 0 after START, the browser could not connect: see the README)")
        if st.button("Begin the interview", type="primary", disabled=not (ctx.state.playing and ok)):
            S.update(steps=build_steps(S.qs), i=0, ans={}, jobs={}, pending={}, pool=ThreadPoolExecutor(2), t_start=time.time(), t_step=time.time(), stage="run"); st.rerun()
    elif S.stage == "run":
        ctx, proc = cam(); st_autorefresh(1000, key="tick"); run(proc)
    else:
        st.success("Interview complete. Thank you!"); report.render(S.rep, candidate=True)
        if st.button("Start a new session"):
            for k in ["qs", "steps", "ans", "jobs", "pending", "eye", "rep", "post", "meta"]: S.pop(k, None)
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
        sub = min(8, int(el // CAL_DOT_S)); proc.label = "calib"; proc.target = GRID[sub]
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
        with st.form(f"qf_{S.i}"):          # nothing is recorded until the small Submit button is pressed; if time runs out the question is skipped
            if q["type"] == "mcq": st.radio("Choose one", [f"{'ABCD'[n]}. {o}" for n, o in enumerate(q["options"])], index=None, key=key)
            elif q["type"] == "tf": st.radio("True or false?", ["True", "False"], index=None, horizontal=True, key=key)
            else: st.text_input("Answer in at most 20 words", max_chars=160, key=key)
            st.form_submit_button("Submit", on_click=advance)
        st.caption("The paragraph is no longer shown. Press **Submit** to lock in your answer; if you do not, the question is skipped when the time ends.")
    elif k == "post":
        proc.label = "post"; st.subheader("Post-task questions")
        with st.form("post_form"):
            tl = [st.slider(n, 0, 100, 50) for n in ["Mental demand", "Physical demand", "Temporal demand", "Performance (100 = poor)", "Effort", "Frustration"]]
            strain = st.slider("Eye strain right now (0 = none, 10 = severe)", 0, 10, 2)
            if st.form_submit_button("Submit", type="primary"): S.post = dict(tlx=round(float(np.mean(tl)), 1), strain=strain); advance(); st.rerun()
    else:  # wrap
        proc.label = "post"
        with st.spinner("Interview over: grading your answers and analysing eye signals…"): finalize(proc)
        st.rerun()


# ------------------------------------------------------------------ model setup / diagnostics
NOTES = {"Pollinations (free, no key)": "Hosted free model, **no key**. Best effort (availability can change). Fine for generating items; it is not used for candidate data unless you choose it as the grader.",
         "Local model (free, GPT4All)": "Runs inside this app on CPU, **no key**. Downloads automatically the first time it is used (about 0.7 GB for the 1B model). Use *List available models* to see other files (for example Llama-3.2-3B-Instruct-Q4_0.gguf on a bigger PC).",
         "Auto (try the free options)": "Tries, in order: providers whose key you added, then Pollinations (no key), then the local model. Nothing to configure.",
         "Groq (Llama, free tier)": "Free key: **console.groq.com**. Llama models, fast.", "Cerebras (Llama, free tier)": "Free key: **cloud.cerebras.ai**.",
         "Google Gemini (free tier)": "Free key: **aistudio.google.com/apikey**.", "Mistral (free tier)": "Free key: **console.mistral.ai** (free experiment plan).",
         "GitHub Models (free)": "Use a GitHub personal access token (free, rate limited).", "OpenRouter (free models)": "Free key: **openrouter.ai/keys**; pick a model ending in `:free`.",
         "Hugging Face": "Token: **huggingface.co/settings/tokens** (Inference Providers).", "Custom": "Any OpenAI-compatible server."}

def api_setup():
    names = list(llm.PROVIDERS); prov = st.selectbox("Free LLM provider", names, index=names.index(llm.CFG["provider"]))
    if prov != llm.CFG["provider"]: llm.CFG["provider"] = prov; S.model = llm.PROVIDERS[prov]["model"]
    p = llm.PROVIDERS[prov]; st.markdown(NOTES.get(prov, ""))
    if prov == "Custom": p["base"] = st.text_input("Base URL (OpenAI-compatible, ending in /v1)", p["base"])
    if llm.needs_key(prov):
        key = st.text_input(f"API key ({p['key']})", value="", type="password", help=f"Or add {p['key']} to Streamlit secrets.")
        if key: llm.CFG["key"] = key.strip()
    if not p.get("auto"): S.model = st.text_input("Model name", S.model or p["model"]).strip() or p["model"]
    ok = llm.ready(); c = st.columns(3)
    c[0].metric("Provider", prov.split(" (")[0]); c[1].metric("Ready", "✅" if ok else "❌ key missing"); c[2].metric("Model test", S.get("_test", "not run"))
    c1, c2, c3 = st.columns(3)
    if c1.button("List available models", disabled=not ok or bool(p.get("auto"))):
        try: ms = llm.list_models(); st.success(f"{len(ms)} models"); st.code("\n".join(ms))
        except Exception as e: st.error(f"Could not list models: {e}")
    if c2.button("Run JSON test", disabled=not ok):
        with st.spinner("Asking the model (the local model downloads automatically the first time)…"):
            try: r = llm.ask("Reply with JSON only.", 'Return {"ok": true, "reply": "hello"}', 0, S.model, max_new=40); S["_test"] = "✅ works"; st.success("Model works."); st.json(r)
            except Exception as e: S["_test"] = "❌ failed"; st.error(f"Failed: {e}")
    if c3.button("Generate one sample item", disabled=not ok):
        with st.spinner("Generating…"): it = bank.make_item("REST API design", "Data and APIs", S.model)
        if it: st.success(f"Valid item generated ({it['words']} words)"); st.write(it["paragraph"]); st.json(it["questions"])
        else: st.error("No valid item after 5 tries. Try another provider, or a larger model.")

def setup():
    st.title("🛠️ Model setup · free local grader (no API key)")
    st.caption("Short answers are graded by a small free embedding model that downloads automatically on first use (like EasyOCR), plus keyword matching. The question bank is built in, so nothing needs a key.")
    S.grader = st.radio("Short-answer grader", ["semantic", "llm"], index=0 if S.get("grader", "semantic") == "semantic" else 1, horizontal=True,
                        format_func=lambda x: {"semantic": "Free semantic grader (default, no key)", "llm": "LLM through an API key (optional)"}[x])
    c = st.columns(3); started = semantic._M["m"] is not None or semantic._M["err"] is not None
    c[0].metric("Grader engine", semantic.engine() if started else "not loaded yet"); c[1].metric("Question bank", f"{len(bank.load())} items"); c[2].metric("LLM for generation", llm.CFG["provider"].split(" (")[0])
    if st.button("Load the free grading model now"):
        with st.spinner("Loading (first use downloads ~130 MB automatically)…"): eng = semantic.warm()
        (st.success if semantic._M["m"] is not None else st.warning)(f"Engine: {eng}" + ("" if semantic._M["m"] is not None else f" (model unavailable: {semantic._M['err']}; keyword scoring still works)"))
    st.subheader("Try the grader")
    b = bank.load(); qs = [(i["area"], q) for i in b for q in i["questions"] if q["type"] == "short"]
    pick = st.selectbox("Question", range(len(qs)), format_func=lambda n: f"{qs[n][0]}: {qs[n][1]['text']}"); q = qs[pick][1]
    st.caption(f"Model answer: {q['key']}"); ans = st.text_input("Type an answer to test", "")
    if st.button("Grade this answer"):
        r = grader.grade(q, ans, S.grader, S.model); st.metric("Score (0-1)", r["score"] if r.get("score") is not None else "error"); st.json({k: v for k, v in r.items() if k != "score"})
    with st.expander("Free LLMs: generate more items (or grade with an LLM)", expanded=True): api_setup()

# ------------------------------------------------------------------ admin
def admin():
    st.title("🗂️ Item bank (admin)")
    ms = [S.model] if llm.ready() else []
    st.info(f"Item generator: **{llm.CFG['provider']}** (change it in **Model setup**). The 23 built-in items are already approved; generating more is optional.")
    if not ms: st.warning("The selected provider needs an API key. Choose *Auto* or *Pollinations* (no key), or add the key in Model setup.")
    b = bank.load(); cov = bank.coverage(b); st.write("**Approved items per cluster**"); st.bar_chart(pd.Series(cov, name="approved"))
    if b: st.download_button("⬇ Download bank.json (back it up: Streamlit Cloud storage is temporary)", json.dumps(b, indent=1), "bank.json", "application/json")
    up = st.file_uploader("Restore a saved bank.json", type="json")
    if up and st.button("Replace the bank with the uploaded file"): bank.save(json.load(up)); st.rerun()
    with st.expander("Generate new items with a free LLM", expanded=not b):
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
    sel = st.selectbox("Open candidate session", tab.session); rep = next(r for r in reps if r["sid"] == sel)
    st.download_button("⬇ Download this session (JSON)", json.dumps(rep, indent=1), f"{sel}.json", "application/json"); st.divider(); report.render(rep)

mode = st.sidebar.radio("Mode", ["Candidate interview", "Item bank (admin)", "Recruiter dashboard", "Model setup"])
if S.stage != "run": st.sidebar.caption("Grader: " + ("free semantic (no key)" if S.get("grader", "semantic") == "semantic" else f"LLM · {S.model}"))
if mode != "Candidate interview" and S.stage in ("camera", "run"): st.sidebar.warning("An interview is in progress.")
{"Candidate interview": candidate, "Item bank (admin)": admin, "Recruiter dashboard": recruiter, "Model setup": setup}[mode]()
