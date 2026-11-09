// Eye geometry shared by the browser component and the tests. Same formulas as eyes.py (MediaPipe Face Mesh indices).
(function (root) {
  const R = [33, 160, 158, 133, 153, 144], L = [362, 385, 387, 263, 373, 380];
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const ear = (p, i) => (d(p[i[1]], p[i[5]]) + d(p[i[2]], p[i[4]])) / (2 * d(p[i[0]], p[i[3]]));
  function row(p) {                       // p: 478 landmarks {x, y} (iris landmarks 468/473 need the refined model)
    const w = d(p[234], p[454]);
    return { ok: 1, ear: (ear(p, R) + ear(p, L)) / 2,
             ix: (p[468].x - p[33].x) / (p[133].x - p[33].x), iy: (p[468].y - p[159].y) / (p[145].y - p[159].y),
             yaw: (p[1].x - (p[234].x + p[454].x) / 2) / w, pitch: (p[1].y - p[152].y) / d(p[10], p[152]) };
  }
  const api = { ear, row, R, L };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.EYE = api;
})(typeof window !== "undefined" ? window : globalThis);
