"""The app must not crash when OpenCV/libGL is missing:  python tests/test_cam_guard.py"""
import sys, os, subprocess; sys.path.insert(0, os.getcwd())
code = "import sys; sys.modules['cv2'] = None; import eyes, numpy as np; p = eyes.EyeProcessor(); print('eyes imports and runs without cv2:', p.fm is not None)"
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True); assert "without cv2: True" in r.stdout, r.stdout + r.stderr[-400:]
print(r.stdout.strip())
code2 = """import sys, numpy as np, av; import eyes
assert not hasattr(sys.modules['cv2'], '__file__'), 'real OpenCV must never be loaded'
p = eyes.EyeProcessor(); fr = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), np.uint8), format='rgb24')
assert p.recv(fr) is fr and len(p.df()) == 1 and p.received == 1
p.fm.process = lambda img: (_ for _ in ()).throw(RuntimeError('boom'))          # a failing frame must not kill the stream
assert p.recv(fr) is fr and p.errors == 1 and len(p.df()) == 2 and p.df().ok.sum() == 0
p.release(); assert p.recv(fr) is fr and len(p.df()) == 2 and p.closed          # after release frames are passed through untouched
print('EyeProcessor hardened: no real cv2, frame errors survived, release works')"""
r2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True); assert "hardened" in r2.stdout, r2.stdout + r2.stderr[-500:]
print(r2.stdout.strip())
import semantic; semantic._M.update(m=None, err="offline"); sys.modules["eyes"] = None          # force the import error
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run(); at.checkbox[0].check(); at.run()
[b for b in at.button if "Continue" in b.label][0].click().run()
assert not at.exception, at.exception
assert any("eye-measurement library could not be loaded" in e.value and "packages.txt" in e.value for e in at.error), [e.value for e in at.error]
print("Import failure shows a clear fix message instead of a crash"); print("PASSED")
