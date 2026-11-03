# ScreenFit – Streamlit app (Llama via Ollama + webcam eye screening)

## Run
```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10-3.12
pip install -r requirements.txt
# Llama locally (https://ollama.com)
ollama pull llama3.1:8b && ollama serve                  # or set OLLAMA_MODEL / OLLAMA_HOST
streamlit run app.py
```
1. **Item bank (admin)** → Generate (Llama writes 60-80 word paragraphs + 5 quiz questions per area) → tick *approved* → Save.
2. **Candidate interview** → pre-session form → camera check (sidebar START) → 20-minute interview → dashboard.
3. **Recruiter dashboard** → cohort overview + panels A-F for any saved session (`data/sessions/*.json`).

Paragraph plan: P1 = Programming or Data and APIs · P2 = Data science, AI and ML · P3 = Web and security · P4 = Systems · P5 = Engineering practice; 5 questions each (2 MCQ, 1 true/false, 2 short answers) = 25.
Timing (1200 s): set-up 60 · calibration 90 · baseline 60 · 5 × (read 35 + 20/20/20/35/35) · post-task 105 · wrap-up 60.

## Tests (no Ollama or webcam needed)
```bash
python tests/test_pipeline.py && python tests/test_run_stage.py
```
## Notes
* Use `localhost` (or HTTPS) so the browser allows the camera. Video is processed in memory; only numeric signals are kept.
* CVS-Q and OSDI totals are entered from the official questionnaires. Fatigue levels use published cut-offs; this is a screening indicator, not a diagnosis.
* Set `SCREEN_DEG` in `features.py` to your screen's visual angle for accurate gaze error in degrees.
