"""Chooses the short-answer grader: free semantic grader (default, no key) or an LLM API (optional)."""
import llm, semantic

def grade(q, answer, mode="semantic", model=None):
    if mode == "llm" and llm.api_key():
        r = llm.grade_short(q, answer, 3, model)
        if r.get("score") is not None: return r
    return semantic.grade_short(q, answer)
