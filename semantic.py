"""Free, key-less short-answer grader: small embedding model (auto-downloads like EasyOCR) + keyword coverage.
If the model cannot be downloaded, it falls back to keyword/word-overlap scoring. Thresholds are heuristic: calibrate against human graders."""
import re, math, numpy as np
_M = {"m": None, "err": None}

def _embedder():
    if _M["m"] is None and _M["err"] is None:
        try:
            from fastembed import TextEmbedding
            _M["m"] = TextEmbedding("BAAI/bge-small-en-v1.5")        # ~130 MB, public, no token
        except Exception as e: _M["err"] = str(e)[:200]
    return _M["m"]

def warm():
    """Download/load the model now (before the interview). Returns the engine name."""
    m = _embedder()
    if m is not None:
        try: list(m.embed(["warm up"]))
        except Exception as e: _M["m"], _M["err"] = None, str(e)[:200]
    return engine()

def engine(): return "embedding model (bge-small)" if _M["m"] is not None else "keyword fallback"

def _cos(a, b):
    v = np.array(list(_M["m"].embed([a, b])), dtype=float); v /= np.linalg.norm(v, axis=1, keepdims=True); return float(v[0] @ v[1])
def _words(t): return set(re.findall(r"[a-z]{3,}", t.lower()))

def grade_short(q, answer, *_, **__):
    a = (answer or "").strip()
    if not a: return {"score": 0.0, "empty": True, "runs": 0, "agree": True}
    kws = [k.lower() for k in q.get("keywords", [])]; low = a.lower()
    hit = sum(1 for k in kws if k in low); need = max(1, math.ceil(len(kws) / 3))
    kw = min(1.0, hit / need) if kws else None
    key = q.get("key", "")
    if _embedder() is not None:
        try: sim = _cos(a, key); s = min(1.0, max(0.0, (sim - 0.55) / 0.30)); how = f"similarity {sim:.2f}"
        except Exception: _M["m"], _M["err"] = None, "runtime error"; return grade_short(q, answer)
    else:
        ov = len(_words(a) & _words(key)) / max(1, len(_words(key))); s = min(1.0, ov * 1.5); how = f"word overlap {ov:.2f}"
    score = s if kw is None else 0.5 * s + 0.5 * kw
    return {"score": round(score, 3), "correctness": int(round(2 * s)), "completeness": int(round(2 * (kw if kw is not None else s))), "runs": 1,
            "agree": True, "why": f"{how}; keywords {hit}/{len(kws)}", "model": engine()}
