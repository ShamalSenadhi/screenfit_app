"""Item bank: 23 areas / 6 clusters, LLM generation, validation, random assembly."""
import os, json, random
from llm import ask
try: import textstat
except ImportError: textstat = None

CLUSTERS = {
 "Programming": ["Data structures and algorithms", "OOP and design patterns", "Clean code, refactoring and debugging"],
 "Data and APIs": ["Relational databases and SQL", "NoSQL and data modelling", "REST API design", "Data engineering and pipelines"],
 "Data science, AI and ML": ["Machine learning fundamentals", "Data preparation and model evaluation", "Deep learning and neural networks", "Generative AI and LLM applications", "MLOps and responsible AI"],
 "Web and security": ["Web and browser fundamentals", "Web security and authentication", "Computer networking"],
 "Systems": ["Concurrency and multithreading", "Operating systems and memory", "Cloud and containers", "CI/CD and DevOps"],
 "Engineering practice": ["Software testing and quality", "Version control and collaboration", "Software architecture and scalability", "Requirements, Agile and SDLC"],
}
# paragraph plan: para 1 draws from Programming OR Data and APIs; paras 2-5 fixed
PLAN = [["Programming", "Data and APIs"], ["Data science, AI and ML"], ["Web and security"], ["Systems"], ["Engineering practice"]]
BANK = "data/bank.json"
SYS = "You write short quiz items for software developer interviews. Reply with JSON only."
EXAMPLE = {"paragraph": "60-80 words", "questions": [
  {"type": "mcq", "text": "...", "options": ["a", "b", "c", "d"], "answer": 0},
  {"type": "mcq", "text": "...", "options": ["a", "b", "c", "d"], "answer": 2},
  {"type": "tf", "text": "...", "answer": True},
  {"type": "short", "text": "...", "key": "model answer in under 20 words", "rubric": "what must be mentioned"},
  {"type": "short", "text": "...", "key": "...", "rubric": "..."}]}
PROMPT = ("Area: {area}. Audience: mid-level software developer.\nWrite ONE paragraph of 60-80 words explaining a concept in this area, "
          "then exactly 5 questions about it, in this order: 2 multiple-choice (4 options, 'answer' = index 0-3), 1 true/false, "
          "2 short-answer (answerable in at most 20 words, with 'key' and 'rubric').\nReturn JSON in exactly this shape: " + json.dumps(EXAMPLE))

def valid(d):
    try:
        qs = d["questions"]; w = len(d["paragraph"].split())
        return (50 <= w <= 95 and [q["type"] for q in qs] == ["mcq", "mcq", "tf", "short", "short"]
                and all(len(q["options"]) == 4 and q["answer"] in range(4) for q in qs[:2])
                and isinstance(qs[2]["answer"], bool) and all(q["key"] for q in qs[3:]))
    except Exception: return False

def make_item(area, cluster, model=None, tries=5):
    for _ in range(tries):
        try: d = ask(SYS, PROMPT.replace("{area}", area), 0.5, model)
        except Exception: continue
        if valid(d):
            d.update(area=area, cluster=cluster, words=len(d["paragraph"].split()), approved=False, id=f"{area[:5]}-{random.randrange(10**6)}",
                     fk=round(textstat.flesch_kincaid_grade(d["paragraph"]), 1) if textstat else None)
            return d

def load():
    """Saved bank if present, otherwise the built-in curated seed bank (23 items, no model needed)."""
    if os.path.exists(BANK): return json.load(open(BANK))
    from seed_bank import items; return items()
def save(b): os.makedirs("data", exist_ok=True); json.dump(b, open(BANK, "w"), indent=1)

def coverage(bank):
    return {c: sum(1 for i in bank if i["cluster"] == c and i["approved"]) for c in CLUSTERS}

def assemble(bank, seed):
    """One item per paragraph slot; random cluster/area/item and shuffled MCQ options. Logged by seed."""
    rng = random.Random(seed); out = []; used = set()
    for opts in PLAN:
        ok = [c for c in opts if any(i["cluster"] == c and i["approved"] for i in bank)]
        if not ok: raise ValueError(f"No approved items for {opts}")
        cl = rng.choice(ok)
        pool = [i for i in bank if i["cluster"] == cl and i["approved"] and i["id"] not in used]
        it = json.loads(json.dumps(rng.choice(pool))); used.add(it["id"])
        for q in it["questions"][:2]:
            correct = q["options"][q["answer"]]; rng.shuffle(q["options"]); q["answer"] = q["options"].index(correct)
        out.append(it)
    return out
