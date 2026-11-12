"""Exports for one session (PDF report + CSV files + ZIP pack) and for the whole cohort (CSV). Numbers only: no video is ever stored."""
import io, math, zipfile
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.lineplots import LinePlot

PROFILE = [("name", "Full name / ID"), ("age", "Age"), ("gender", "Gender"), ("email", "Email"), ("phone", "Phone"),
           ("qualification", "Highest qualification"), ("field", "Field of study"), ("institution", "Institution"),
           ("experience_years", "Years of experience"), ("job_title", "Current / last job title"), ("level", "Role level applied for"),
           ("skills", "Main skills / languages"), ("certifications", "Certifications"), ("glasses", "Wears glasses"), ("lenses", "Wears contact lenses"),
           ("eye_conditions", "Eye conditions"), ("last_eye_exam", "Last eye examination"), ("other_medical", "Other medical conditions"),
           ("medications", "Medication (vision / alertness)"), ("sleep_hours", "Hours slept last night"), ("screen_hours", "Screen hours per day"),
           ("cvsq", "CVS-Q total score"), ("osdi", "OSDI total score")]
DISCLAIMER = ("This is a screening indicator of visual-fatigue risk, not a medical diagnosis. It is never used to rank or reject candidates, "
              "and a human must review every result. If risk is moderate or high, an eye check-up is suggested.")
REC = {"Low": ["No special action needed.", "Keep regular screen breaks (20-20-20 rule)."],
       "Moderate": ["Take a short break every 20 minutes.", "Adjust font size, brightness and screen distance.", "Consider an eye check-up if symptoms persist."],
       "High": ["Take frequent short breaks and use larger fonts.", "An eye check-up is recommended.", "Ask the employer about ergonomic accommodations."]}
RISK_COL = {"Low": "#8BC34A", "Moderate": "#FFC107", "High": "#E53935"}

def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or d.get(k) is None: return default
        d = d[k]
    return d

def fmt(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "-"
    if isinstance(v, bool): return "Yes" if v else "No"
    if isinstance(v, (list, tuple)): return "; ".join(map(str, v)) if v else "-"
    if isinstance(v, float): return f"{v:.{d}f}"
    return str(v) if str(v).strip() else "-"

def _name(rep): return g(rep, "meta", "name", default=rep.get("sid", ""))

# ------------------------------------------------------------------ CSV
def summary_row(rep):
    m, t, v = rep.get("meta", {}), rep["technical"], rep["visual"]; e = v.get("ear") or {}; fr, st_ = e.get("frames") or {}, e.get("stats") or {}; ev = e.get("events") or []
    mins = max((fr.get("duration_s") or 0) / 60, 1e-9); q = pd.DataFrame(t["questions"])
    row = {"session": rep["sid"], "date": rep["when"], "item_set_seed": rep.get("seed")}
    row.update({k: (", ".join(m[k]) if isinstance(m.get(k), list) else m.get(k)) for k, _ in PROFILE})
    row.update(technical_score=round(t["overall"], 1), visual_risk=v["risk"], supporting_eye_flags="; ".join(v.get("flags") or []),
               nasa_tlx=g(rep, "post", "tlx"), strain_0_10=g(rep, "post", "strain"), grader=m.get("grader"),
               timed_out_questions=int(q.timed_out.sum()) if "timed_out" in q else None, capture=m.get("capture"))
    row.update({f"score_{k}": round(x, 1) for k, x in (t.get("clusters") or {}).items()})
    row.update({f"score_type_{k}": round(x, 1) for k, x in (t.get("by_type") or {}).items()})
    row.update(frames_total=fr.get("total"), face_detected_pct=fr.get("face_pct"), frame_rate=fr.get("fps"), duration_s=fr.get("duration_s"),
               baseline_ear=st_.get("ear0"), blink_threshold=st_.get("thr_blink"), closure_threshold=st_.get("thr_close"), ear_mean=st_.get("ear_mean"), ear_sd=st_.get("ear_std"), ear_min=st_.get("ear_min"),
               eye_closed_frames=fr.get("closed"), blinks_total=len(ev) if e else None, blink_rate_per_min=(len(ev) / mins) if e else None,
               mean_blink_ms=(sum(x["dur_ms"] for x in ev) / len(ev)) if ev else None, gaze_error_deg=g(v, "quality", "gaze_err_deg"))
    return row

def summary_df(reps): return pd.DataFrame([summary_row(r) for r in reps])
def _tag(df, rep): return df.assign(session=rep["sid"], name=_name(rep)).pipe(lambda d: d[["session", "name"] + [c for c in d.columns if c not in ("session", "name")]])
def questions_df(rep): return _tag(pd.DataFrame(rep["technical"]["questions"]), rep)
def paragraph_df(rep): return _tag(pd.DataFrame(g(rep, "visual", "para", default=[])), rep)
def ear_df(rep): return _tag(pd.DataFrame(g(rep, "visual", "ear", "series", default=[]), columns=["t", "ear", "step"]), rep)
def events_df(rep): return _tag(pd.DataFrame(g(rep, "visual", "ear", "events", default=[]), columns=["t", "dur_ms", "min_ear", "step"]), rep)
def csv_bytes(df): return df.to_csv(index=False).encode("utf-8-sig")        # BOM so Excel shows non-English names correctly

def zip_bytes(rep, pdf=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", csv_bytes(summary_df([rep]))); z.writestr("questions.csv", csv_bytes(questions_df(rep)))
        z.writestr("eye_by_paragraph.csv", csv_bytes(paragraph_df(rep))); z.writestr("ear_series.csv", csv_bytes(ear_df(rep)))
        z.writestr("blink_events.csv", csv_bytes(events_df(rep))); z.writestr("report.pdf", pdf if pdf is not None else pdf_bytes(rep))
    return buf.getvalue()

# ------------------------------------------------------------------ PDF
def T(x):
    """Latin-1 safe, XML-escaped text (the built-in PDF fonts cannot show other scripts; the CSV files keep full Unicode)."""
    return escape(str(x)).encode("latin-1", "replace").decode("latin-1")

ST = getSampleStyleSheet()
BASE = ParagraphStyle("b", parent=ST["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11)
SMALL = ParagraphStyle("s", parent=BASE, fontSize=7.6, leading=9.6, textColor=colors.HexColor("#555555"))
BIG = ParagraphStyle("big", parent=BASE, leading=26, spaceBefore=2)
H1 = ParagraphStyle("h1", parent=ST["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#1F3864"), alignment=0, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=ST["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2F5597"), spaceBefore=10, spaceAfter=4)
NAVY, GRID, HEAD = colors.HexColor("#2F5597"), colors.HexColor("#C9D1E0"), colors.HexColor("#E2EBF8")

def P(x, s=BASE): return Paragraph(T(x), s)
def M(markup, s=BASE): return Paragraph(markup, s)                  # markup built from numbers / T()-escaped text only
def tbl(rows, widths, head=True, font=8, zebra=True):
    data = [[c if not isinstance(c, str) else P(c, ParagraphStyle("c", parent=BASE, fontSize=font, leading=font + 2.4, fontName="Helvetica-Bold" if (head and i == 0) else "Helvetica")) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head: style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    if zebra: style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7F9FD")) for i in range(2 if head else 1, len(rows), 2)]
    t.setStyle(TableStyle(style)); return t

def bar_chart(items, w):
    h = 12 * mm + 8 * mm * max(1, len(items)); d = Drawing(w, h); c = HorizontalBarChart()
    c.x, c.y, c.width, c.height = 46 * mm, 5 * mm, w - 56 * mm, h - 8 * mm
    c.data = [[float(v) for _, v in items][::-1]]; c.categoryAxis.categoryNames = [T(k) for k, _ in items][::-1]
    c.valueAxis.valueMin, c.valueAxis.valueMax, c.valueAxis.valueStep = 0, 100, 20
    c.bars[0].fillColor = NAVY; c.barLabelFormat = "%d"; c.barLabels.fontSize = 7.5; c.barLabels.nudge = 7
    c.categoryAxis.labels.fontSize = 8; c.valueAxis.labels.fontSize = 7.5; c.categoryAxis.labels.fontName = c.valueAxis.labels.fontName = c.barLabels.fontName = "Helvetica"; c.categoryAxis.labels.boxAnchor = "e"; c.bars.strokeColor = None
    d.add(c); return d

def line_chart(series, labels, cols, w, h, ymin=None, ymax=None, dashed=()):
    d = Drawing(w, h); lp = LinePlot(); lp.x, lp.y, lp.width, lp.height = 11 * mm, 7 * mm, w - 16 * mm, h - 15 * mm
    lp.data = series
    for i, col in enumerate(cols):
        lp.lines[i].strokeColor = colors.HexColor(col); lp.lines[i].strokeWidth = 1.1; lp.lines[i].symbol = None
        if i in dashed: lp.lines[i].strokeDashArray = (3, 2)
    if ymin is not None: lp.yValueAxis.valueMin = ymin
    if ymax is not None: lp.yValueAxis.valueMax = ymax
    lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7; lp.xValueAxis.labels.fontName = lp.yValueAxis.labels.fontName = "Helvetica"; lp.xValueAxis.visibleGrid = lp.yValueAxis.visibleGrid = 1
    lp.xValueAxis.gridStrokeColor = lp.yValueAxis.gridStrokeColor = colors.HexColor("#E3E7EF")
    d.add(lp)
    for i, (lab, col) in enumerate(zip(labels, cols)):
        d.add(Line(12 * mm + i * 42 * mm, h - 4 * mm, 17 * mm + i * 42 * mm, h - 4 * mm, strokeColor=colors.HexColor(col), strokeWidth=1.6)); d.add(String(18 * mm + i * 42 * mm, h - 6 * mm, T(lab), fontSize=7))
    return d

def _binned(series, n=380):
    if len(series) <= n: return [(float(s["t"]), float(s["ear"])) for s in series]
    k = math.ceil(len(series) / n); return [(float(series[i]["t"]), min(float(x["ear"]) for x in series[i:i + k])) for i in range(0, len(series), k)]   # keep blink dips

def pdf_bytes(rep):
    m, t, v = rep.get("meta", {}), rep["technical"], rep["visual"]; e = v.get("ear") or {}; fr, s_ = e.get("frames") or {}, e.get("stats") or {}; ev = e.get("events") or []
    q = pd.DataFrame(t["questions"]); vp = pd.DataFrame(v.get("para") or []); ph = pd.DataFrame(e.get("phases") or []); qual = v.get("quality") or {}; W = 174 * mm; S = []
    S += [P("ScreenFit - Candidate Screening Report", H1), P(f"Session {rep['sid']}  |  {rep['when']}  |  item-set seed {rep.get('seed')}  |  grader: {m.get('grader', '-')}", SMALL), Spacer(1, 4 * mm)]
    risk = v["risk"]; face = f"{qual['face_ok']:.0f}%" if qual.get("face_ok") is not None else "-"
    top = Table([[P("Technical score", SMALL), P("Visual fitness (screening)", SMALL), P("Face detected", SMALL), P("Timed-out questions", SMALL)],
                 [M(f"<font size=20><b>{t['overall']:.0f}</b></font> / 100", BIG), M(f"<font size=14><b>{T(risk)} risk</b></font>", BIG), M(f"<font size=14><b>{T(face)}</b></font>", BIG), M(f"<font size=14><b>{int(q.timed_out.sum()) if 'timed_out' in q else 0}</b></font>", BIG)]],
                colWidths=[W / 4] * 4)
    top.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, NAVY), ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID), ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("BACKGROUND", (1, 1), (1, 1), colors.HexColor(RISK_COL[risk])),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 1), (-1, 1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 6)]))
    S += [top, Spacer(1, 1.5 * mm), P("Panels A-B (technical) and C (visual fitness) are reported separately and are never combined into one ranking score.", SMALL)]
    S += [P("Candidate profile", H2)]
    pr = [(lab, fmt(m.get(k))) for k, lab in PROFILE]; half = math.ceil(len(pr) / 2); rows = []
    for i in range(half): rows.append([pr[i][0], pr[i][1]] + (list(pr[i + half]) if i + half < len(pr) else ["", ""]))
    pt = tbl(rows, [34 * mm, 53 * mm, 34 * mm, 53 * mm], head=False, font=7.6, zebra=False)
    pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), HEAD), ("BACKGROUND", (2, 0), (2, -1), HEAD)])); S.append(pt)
    S += [P("A-B  Technical interview and skills profile", H2),
          M(f"Overall technical score: <b>{t['overall']:.0f} / 100</b>. There are 25 questions (per paragraph: 2 multiple-choice, 1 true/false and 2 short answers). Multiple-choice and true/false items are scored automatically; "
            f"short answers are graded by {T(m.get('grader', 'the grader'))}. The score uses answers only: no eye or health data is used.")]
    cl = list((t.get("clusters") or {}).items())
    if cl: S += [bar_chart(cl, W), Spacer(1, 3 * mm)]
    S += [tbl([["Question type", "Score"]] + [[k, f"{x:.0f}"] for k, x in (t.get("by_type") or {}).items()], [60 * mm, 30 * mm])]
    S += [P("C  Visual fitness indicator (screening only)", H2), tbl([[P(DISCLAIMER, ParagraphStyle("d", parent=BASE, textColor=colors.HexColor("#7A5200")))]], [W], head=False, zebra=False)]
    S[-1].setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF3D6")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E6C36A"))]))
    S += [Spacer(1, 2 * mm), tbl([["Instrument", "Value", "Published cut-off"], ["CVS-Q", fmt(m.get("cvsq")), ">= 6: symptoms present"], ["OSDI", fmt(m.get("osdi")), "0-12 normal, 13-22 mild, 23-32 moderate, 33-100 severe"],
                                  ["Rule outcome", f"{risk} risk", "Low / Moderate / High (CVS-Q and OSDI rule)"]], [35 * mm, 35 * mm, 104 * mm])]
    if m.get("eye_conditions") or m.get("other_medical") or m.get("glasses") or m.get("lenses"):
        S += [P(f"Declared context: glasses {fmt(m.get('glasses'))}, contact lenses {fmt(m.get('lenses'))}, eye conditions {fmt(m.get('eye_conditions'))}, other conditions {fmt(m.get('other_medical'))}. "
                "These can influence eye measurements and are shown for interpretation only.", SMALL)]
    flags = v.get("flags"); S += [P("Supporting eye indicators (rule baseline, not part of the level): " + ("no usable eye data." if flags is None else ("no fatigue-like pattern detected." if not flags else "; ".join(flags))), BASE)]
    S += [P("Recommendation: " + " ".join(REC[risk]), BASE)]
    S += [P("D-E  Fatigue trend and reading behaviour", H2)]
    if not vp.empty and vp.blink_rate.notna().any():
        x = [int(p) for p in vp.para]; S += [line_chart([list(zip(x, vp.blink_rate.fillna(0))), list(zip(x, vp.perclos.fillna(0)))], ["Blinks per minute", "PERCLOS %"], ["#2F5597", "#E07B24"], W, 52 * mm, ymin=0)]
        S += [tbl([["Paragraph", "Blinks/min", "PERCLOS %", "Incomplete blinks", "Regression %", "Fixation ms", "Gaze-away %"]] +
                  [[str(int(r.para)), fmt(r.blink_rate), fmt(r.perclos), fmt(100 * r.incomplete if r.incomplete == r.incomplete else None, 0) + ("%" if r.incomplete == r.incomplete else ""),
                    fmt(getattr(r, "regress", None)), fmt(getattr(r, "fix_ms", None), 0), fmt(getattr(r, "away", None))] for r in vp.itertuples()], [24 * mm, 24 * mm, 24 * mm, 30 * mm, 24 * mm, 24 * mm, 24 * mm])]
    else: S += [P("No usable eye data was recorded.", BASE)]
    S += [P("EAR and frame data", H2), P("No video is recorded or stored. Only the eye aspect ratio (EAR) of each analysed frame is kept.", SMALL)]
    if e:
        mins = max(fr.get("duration_s", 0) / 60, 1e-9); dur = fr.get("duration_s", 0)
        S += [tbl([["Measure", "Value", "Measure", "Value"],
                   ["Frames analysed", f"{fr.get('total', 0):,}", "Baseline EAR", fmt(s_.get("ear0"), 3)],
                   ["Face detected", f"{fmt(fr.get('face_pct'))}%", "Blink threshold (72%)", fmt(s_.get("thr_blink"), 3)],
                   ["Frame rate", f"{fmt(fr.get('fps'), 0)} fps", "Closure threshold (20%)", fmt(s_.get("thr_close"), 3)],
                   ["Duration", f"{int(dur // 60)}:{int(dur % 60):02d}", "Eye-closed frames", f"{fr.get('closed', 0):,}"],
                   ["Blinks (total)", str(len(ev)), "EAR mean +/- SD", f"{fmt(s_.get('ear_mean'), 3)} +/- {fmt(s_.get('ear_std'), 3)}"],
                   ["Blink rate", f"{len(ev) / mins:.1f} /min", "Mean blink duration", f"{sum(x['dur_ms'] for x in ev) / len(ev):.0f} ms" if ev else "-"]], [40 * mm, 47 * mm, 40 * mm, 47 * mm])]
        ser = e.get("series") or []
        if ser:
            pts = _binned(ser); x0, x1 = pts[0][0], pts[-1][0]
            S += [Spacer(1, 2 * mm), P("EAR over the session (lowest EAR per 0.25 s window, so blinks appear as dips; dashed lines are the blink and closure thresholds)", SMALL),
                  line_chart([pts, [(x0, s_["thr_blink"]), (x1, s_["thr_blink"])], [(x0, s_["thr_close"]), (x1, s_["thr_close"])]], ["EAR", "Blink threshold", "Closure threshold"], ["#2F5597", "#B08900", "#C00000"], W, 56 * mm, ymin=0, dashed=(1, 2))]
        if not ph.empty:
            S += [Spacer(1, 2 * mm), tbl([["Phase", "Frames", "Face frames", "EAR mean", "EAR min", "Blinks", "Blinks/min", "PERCLOS %"]] +
                  [[r.phase, str(int(r.frames)), str(int(r.face_frames)), fmt(r.ear_mean, 3), fmt(r.ear_min, 3), str(int(r.blinks)), fmt(r.blink_rate), fmt(r.perclos)] for r in ph.itertuples()], [26 * mm, 21 * mm, 23 * mm, 22 * mm, 22 * mm, 18 * mm, 22 * mm, 20 * mm])]
    S += [P("F  Data quality and safeguards", H2),
          tbl([["Check", "Value", "Target"], ["Face detected (% of frames)", fmt(qual.get("face_ok")), ">= 90"], ["Gaze calibration error (deg)", fmt(qual.get("gaze_err_deg")), "<= 4"],
               ["Grader failures", str(int(t["grader"]["failed"])) if t.get("grader") else "-", "0"], ["Empty short answers", str(int(t["grader"]["empty"])) if t.get("grader") else "-", "-"]], [70 * mm, 50 * mm, 54 * mm]),
          Spacer(1, 1.5 * mm), P("Safeguards: video processed on the device (or in memory) and never stored; only derived numbers are saved; technical and visual panels are never combined; human review is required before any use.", SMALL)]
    S += [P("Questions and answers", H2)]
    rows = [["P", "Q", "Type", "Area", "Answer", "Score", "Note"]]
    for r in q.itertuples():
        ans = (str(r.answer) if r.answer is not None else "")[:60]; sc = "-" if r.score is None or r.score != r.score else f"{100 * r.score:.0f}"
        rows.append([str(r.para), str(r.q), r.type, r.area, ans or "-", sc, "timed out" if r.timed_out else ""])
    S += [tbl(rows, [8 * mm, 8 * mm, 14 * mm, 46 * mm, 56 * mm, 14 * mm, 28 * mm], font=7)]

    def footer(c, doc):
        c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor("#777777"))
        c.drawString(18 * mm, 10 * mm, T(f"ScreenFit report | session {rep['sid']} | confidential: contains personal and optional health details | screening indicator only"))
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}"); c.restoreState()
    buf = io.BytesIO(); SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=17 * mm, title=f"ScreenFit report {rep['sid']}", author="ScreenFit").build(S, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
