"""Scoring + the full Streamlit dashboard (panels A-F). Technical and visual results are never combined."""
import math, time, numpy as np, pandas as pd, streamlit as st

def risk_level(cvsq, osdi):
    """Published cut-offs: CVS-Q >= 6 (Segui et al.); OSDI bands 0-12 / 13-22 / 23-32 / 33-100 (Schiffman et al.)."""
    if cvsq >= 6 and osdi >= 23: return "High"
    if cvsq >= 6 or osdi >= 13: return "Moderate"
    return "Low"

def eye_signs(f):
    """Supporting rule baseline (tune in the pilot). Returns a list of flags, or None when no usable eye data."""
    if not f: return None
    p, b, flags = f["para"], f["baseline"], []
    br, pc, inc = p.blink_rate.dropna(), p.perclos.dropna(), p.incomplete.dropna()
    if len(br) and b["blink_rate"] == b["blink_rate"] and br.mean() < 0.7 * b["blink_rate"]: flags.append("Blink rate more than 30% below personal baseline")
    if len(pc) >= 4 and pc.iloc[-2:].mean() > 1.5 * max(pc.iloc[:2].mean(), 0.5): flags.append("PERCLOS rising across the session")
    if len(inc) >= 4 and inc.iloc[-2:].mean() - inc.iloc[:2].mean() > 0.10: flags.append("More incomplete blinks late in the session")
    return flags

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame): return clean(o.to_dict("records"))
    if isinstance(o, (np.floating, float)): return None if (o != o or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o

def build(sid, seed, meta, items, ans, grades, feats, post):
    rows = []
    for pi, it in enumerate(items):
        for qi, q in enumerate(it["questions"]):
            k = f"{pi}-{qi}"; a = ans.get(k, {}); v = a.get("val"); g = grades.get(k)
            if q["type"] in ("mcq", "tf"): s = 1.0 if (v is not None and v == q["answer"]) else 0.0
            else: s = None if g is None else g.get("score")
            shown = "" if v is None else ("ABCD"[v] if q["type"] == "mcq" else str(v))
            rows.append(dict(para=pi + 1, q=qi + 1, type=q["type"], cluster=it["cluster"], area=it["area"], question=q["text"], answer=shown,
                             score=s, timed_out=bool(a.get("timed_out")), why=(g or {}).get("why", ""), agree=(g or {}).get("agree")))
    df = pd.DataFrame(rows)
    sh = df[df.type == "short"]
    tech = dict(overall=float(100 * df.groupby("para").score.mean().mean()),
                clusters=(100 * df.groupby("cluster").score.mean()).to_dict(), by_type=(100 * df.groupby("type").score.mean()).to_dict(),
                para=(100 * df.groupby("para").score.mean()).tolist(), questions=df,
                grader=dict(short=len(sh), empty=int((sh.answer == "").sum()), failed=int(sh.score.isna().sum()),
                            agree=float(sh.agree.dropna().astype(bool).mean() * 100) if sh.agree.notna().any() else None))
    flags = eye_signs(feats)
    vis = dict(risk=risk_level(meta["cvsq"], meta["osdi"]), flags=flags,
               baseline=feats["baseline"] if feats else None, ear0=feats["ear0"] if feats else None,
               para=feats["para"] if feats else pd.DataFrame(), quality=feats["quality"] if feats else {})
    return clean(dict(sid=sid, seed=seed, when=time.strftime("%Y-%m-%d %H:%M"), meta=meta, technical=tech, visual=vis, post=post,
                      items=[dict(para=i + 1, cluster=it["cluster"], area=it["area"], id=it["id"], text=it["paragraph"]) for i, it in enumerate(items)]))

REC = {"Low": ["No special action needed.", "Keep regular screen breaks (20-20-20 rule)."],
       "Moderate": ["Take a short break every 20 minutes.", "Adjust font size, brightness and screen distance.", "Consider an eye check-up if symptoms persist."],
       "High": ["Take frequent short breaks and use larger fonts.", "An eye check-up is recommended.", "Ask the employer about ergonomic accommodations."]}
COL = {"Low": st.success, "Moderate": st.warning, "High": st.error}
DISC = "**Screening disclaimer.** This is a screening indicator of visual-fatigue risk, not a medical diagnosis. It is never used to rank or reject candidates."

def df_(x): return pd.DataFrame(x) if x is not None else pd.DataFrame()

def render(rep, candidate=False):
    t, v, m = rep["technical"], rep["visual"], rep["meta"]; q = df_(t["questions"]); vp = df_(v["para"]); qual = v.get("quality") or {}
    if candidate: st.info("You see your own results first. Technical and visual-fitness results are shown separately.")
    st.caption(f"Session {rep['sid']} · {rep['when']} · item-set seed {rep['seed']} · role level: {m.get('level','-')}")
    k = st.columns(4)
    k[0].metric("Technical score", f"{t['overall']:.0f} / 100"); k[1].metric("Visual fitness", f"{v['risk']} risk")
    k[2].metric("Face detected", f"{qual.get('face_ok', float('nan')):.0f}%" if qual else "n/a"); k[3].metric("Timed-out questions", int(q.timed_out.sum()))
    tabs = st.tabs(["Overview (A-F)", "A Technical score", "B Skills profile", "C Visual fitness", "D Fatigue trend", "E Reading behaviour", "F Quality & safeguards", "Items & grading"])
    with tabs[0]:
        a, b, c = st.columns([1, 1.4, 1.2])
        with a:
            st.subheader("A · Technical score"); st.metric("Overall", f"{t['overall']:.0f}"); st.progress(min(1.0, t["overall"] / 100))
            st.caption("Computed from answers only. No eye or health data is used.")
        with b:
            st.subheader("B · Skills profile"); st.bar_chart(pd.Series(t["clusters"], name="score"))
        with c:
            st.subheader("C · Visual fitness"); COL[v["risk"]](f"{v['risk']} risk"); st.markdown(DISC)
            for r in REC[v["risk"]]: st.write("• " + r)
        d, e, f = st.columns([1.3, 1, 1])
        with d:
            st.subheader("D · Fatigue trend")
            if not vp.empty: st.line_chart(vp.set_index("para")[["blink_rate", "perclos"]])
            else: st.caption("No usable eye data.")
        with e:
            st.subheader("E · Reading behaviour")
            if not vp.empty: st.dataframe(vp[["para", "regress", "fix_ms", "away"]].round(1), hide_index=True, use_container_width=True)
        with f:
            st.subheader("F · Quality & safeguards"); st.write(f"✅ Face detected: {qual.get('face_ok', float('nan')):.0f}% of frames" if qual else "⚠️ No eye data")
            st.write(f"✅ Gaze error: {qual.get('gaze_err_deg', float('nan')):.1f}°" if qual and qual.get("gaze_err_deg") is not None else "⚠️ Gaze calibration unavailable")
            st.write("✅ Raw video never stored"); st.write("⚠️ Panels A-B and C are never combined"); st.write("⚠️ Human review required")
    with tabs[1]:
        st.subheader("A · Technical interview score"); c1, c2 = st.columns(2)
        c1.metric("Overall", f"{t['overall']:.0f} / 100"); c1.bar_chart(pd.Series(t["para"], index=[f"P{i+1}" for i in range(len(t["para"]))], name="score by paragraph"))
        c2.bar_chart(pd.Series(t["by_type"], name="score by question type"))
        st.dataframe(q[["para", "q", "type", "area", "answer", "score", "timed_out"]], hide_index=True, use_container_width=True)
        g = t["grader"]; st.caption(f"Short answers: {g['short']} · empty: {g['empty']} · grader failures: {g['failed']} · runs agreeing: {g['agree'] if g['agree'] is None else round(g['agree'])}%")
    with tabs[2]:
        st.subheader("B · Skills profile (clusters seen)"); st.bar_chart(pd.Series(t["clusters"], name="score"))
        cl = q.groupby("cluster").agg(areas=("area", "first"), score=("score", "mean")).assign(score=lambda x: (100 * x.score).round(0)); st.dataframe(cl, use_container_width=True)
        st.caption("Each candidate sees one area from each of the five paragraph slots (23 areas in 6 clusters overall); the profile never claims knowledge of areas not seen.")
    with tabs[3]:
        st.subheader("C · Visual fitness indicator"); st.warning(DISC); COL[v["risk"]](f"Risk level: {v['risk']}")
        st.table(pd.DataFrame({"Instrument": ["CVS-Q", "OSDI", "Rule outcome"], "Value": [str(m["cvsq"]), str(m["osdi"]), v["risk"]],
                               "Cut-off": ["≥ 6 = symptoms", "0-12 normal · 13-22 mild · 23-32 moderate · 33-100 severe", "Low / Moderate / High"]}))
        st.markdown("**Recommendation**"); [st.write("• " + r) for r in REC[v["risk"]]]
        st.markdown("**Supporting eye indicators (rule baseline, not part of the level)**")
        if v["flags"] is None: st.caption("No usable eye data.")
        elif not v["flags"]: st.write("✅ No fatigue-like pattern detected in the 20-minute session.")
        else: [st.write("⚠️ " + x) for x in v["flags"]]
        st.caption("Not included in the technical score or ranking. Human review required.")
    with tabs[4]:
        st.subheader("D · Fatigue trend across paragraphs")
        if vp.empty: st.caption("No usable eye data.")
        else:
            b = v["baseline"] or {}; c1, c2, c3 = st.columns(3)
            c1.metric("Baseline blink rate", f"{b.get('blink_rate', float('nan')):.1f} /min"); c2.metric("Baseline PERCLOS", f"{b.get('perclos', float('nan')):.1f} %"); c3.metric("Baseline EAR", f"{v['ear0']:.3f}")
            x = vp.set_index("para"); st.line_chart(x[["blink_rate"]].assign(baseline=b.get("blink_rate")))
            c1, c2 = st.columns(2); c1.line_chart(x[["perclos"]]); c2.line_chart(x[["incomplete"]]); st.dataframe(vp.round(2), hide_index=True, use_container_width=True)
        p = rep.get("post") or {}; st.caption(f"Post-task strain rating: {p.get('strain','-')}/10 · NASA-TLX (raw mean): {p.get('tlx','-')}")
    with tabs[5]:
        st.subheader("E · Reading behaviour")
        if vp.empty: st.caption("No usable eye data.")
        else:
            st.dataframe(vp[["para", "regress", "fix_ms", "away"]].rename(columns={"regress": "Regression rate %", "fix_ms": "Mean fixation ms", "away": "Gaze-away %"}).round(1), hide_index=True, use_container_width=True)
            st.line_chart(vp.set_index("para")[["regress"]])
        st.caption("Gaze is analysed at line/paragraph level only; webcam gaze is precise to several degrees.")
    with tabs[6]:
        st.subheader("F · Data quality and safeguards")
        rows = [("Face detected (% of frames)", qual.get("face_ok"), "≥ 90"), ("Gaze calibration error (deg)", qual.get("gaze_err_deg"), "≤ 4"),
                ("Frames recorded", qual.get("frames"), ""), ("Timed-out questions", int(q.timed_out.sum()), "0"), ("Grader failures", t["grader"]["failed"], "0")]
        st.table(pd.DataFrame(rows, columns=["Check", "Value", "Target"]).astype(str))
        for s in ["Video processed locally and never stored", "Only derived numbers saved under an anonymous ID", "Technical (A-B) and visual (C) panels never combined", "Candidate sees own results first", "Human review required before any use"]: st.write("✅ " + s)
    with tabs[7]:
        st.subheader("Items and grading")
        for it in rep["items"]:
            with st.expander(f"Paragraph {it['para']} · {it['cluster']} · {it['area']} ({it['id']})"):
                st.write(it["text"]); d = q[q.para == it["para"]]
                for _, r in d.iterrows(): st.markdown(f"**Q{r.q} ({r.type})** {r.question}  \n→ answer: `{r.answer or '—'}` · score: **{r.score}** {'⏱ timed out' if r.timed_out else ''}  \n_{r.why or ''}_")
