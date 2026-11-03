"""Eye features from per-frame signals (pure numpy/pandas/sklearn; no video)."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupKFold, cross_val_predict
F = ["ix", "iy", "yaw", "pitch"]
SCREEN_DEG = np.array([33.0, 19.0])   # assumed visual angle of a ~15" screen at 60 cm (set from your set-up)

def blinks(d, ear0, lo=0.72, min_ms=70, max_ms=400):
    out, start = [], None; t = d.t.values; e = d.ear.values
    for i in range(len(d)):
        if e[i] < lo * ear0 and start is None: start = i
        if (e[i] >= lo * ear0 or i == len(d) - 1) and start is not None:
            ms = (t[i] - t[start]) * 1000
            if min_ms <= ms <= max_ms: out.append((ms, e[start:i + 1].min()))
            start = None
    return out

def summ(d, ear0):
    d = d[d.ok == 1]
    if len(d) < 10: return dict(blink_rate=np.nan, perclos=np.nan, incomplete=np.nan, blink_ms=np.nan)
    mins = max((d.t.max() - d.t.min()) / 60, 1e-6); b = blinks(d, ear0)
    return dict(blink_rate=len(b) / mins, perclos=float((d.ear < 0.2 * ear0).mean() * 100),
                incomplete=float(np.mean([m > 0.2 * ear0 for _, m in b])) if b else 0.0,
                blink_ms=float(np.mean([x for x, _ in b])) if b else np.nan)

def fit_gaze(df):
    c = df[(df.label == "calib") & (df.ok == 1) & df.tx.notna()].copy()
    if c.empty: return None, np.nan
    c["chg"] = (c[["tx", "ty"]] != c[["tx", "ty"]].shift()).any(axis=1).cumsum()
    c = c[c.t - c.groupby("chg").t.transform("min") > 3]          # drop settling time after each dot
    if len(c) < 60 or c.chg.nunique() < 5: return None, np.nan
    m = make_pipeline(PolynomialFeatures(2), Ridge(1.0)); Y = c[["tx", "ty"]].values
    pred = cross_val_predict(m, c[F], Y, cv=GroupKFold(min(5, c.chg.nunique())), groups=c.chg)
    return m.fit(c[F], Y), float(np.hypot(*(np.abs(pred - Y).mean(0) * SCREEN_DEG)))

def idt(xy, t, disp=0.04, min_s=0.1):
    out, i, n = [], 0, len(xy)
    while i < n:
        j = i
        while j < n and t[j] - t[i] < min_s: j += 1
        if j >= n: break
        ptp = lambda a, b: np.ptp(xy[a:b + 1, 0]) + np.ptp(xy[a:b + 1, 1])
        if ptp(i, j) <= disp:
            while j + 1 < n and ptp(i, j + 1) <= disp: j += 1
            out.append((t[i], t[j] - t[i], xy[i:j + 1].mean(0))); i = j + 1
        else: i += 1
    return out

def reading(d, model):
    d = d[d.ok == 1]
    if model is None or len(d) < 20: return dict(regress=np.nan, fix_ms=np.nan, away=np.nan)
    g = model.predict(d[F]); fx = idt(g, d.t.values)
    reg = sum(1 for a, b in zip(fx, fx[1:]) if (abs(b[2][1] - a[2][1]) < 0.05 and b[2][0] - a[2][0] < -0.06) or b[2][1] - a[2][1] < -0.08)
    return dict(regress=100 * reg / max(1, len(fx) - 1), fix_ms=1000 * float(np.mean([f[1] for f in fx])) if fx else np.nan,
                away=100 * float(((g < -0.1) | (g > 1.1)).any(axis=1).mean()))

def compute(df):
    """Returns dict(ear0, baseline, para DataFrame, quality) or None if baseline is unusable."""
    if df.empty: return None
    b = df[(df.label == "baseline") & (df.ok == 1)]
    if len(b) < 30: return None
    ear0 = float(b.ear.median()); m, err = fit_gaze(df); rows = []
    for p in range(1, 6):
        d = df[df.label.isin([f"read{p}", f"q{p}"])]
        rows.append(dict(para=p, **summ(d, ear0), **reading(d[d.label == f"read{p}"], m)))
    run = df[~df.label.isin(["idle", "setup"])]
    return dict(ear0=ear0, baseline=summ(b, ear0), para=pd.DataFrame(rows),
                quality=dict(face_ok=float(run.ok.mean() * 100), gaze_err_deg=err, frames=int(len(df))))
