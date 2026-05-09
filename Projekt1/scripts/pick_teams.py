"""Interaktiver Team-Farb-Picker — läuft im Browser (kein OpenCV-Fenster nötig).

Zeigt das erste Frame mit YOLO-Detektionen im Browser.
Der User klickt auf einen Spieler von Team A und einen von Team B.
Die HSV-Referenzfarben werden in team_colors.json gespeichert.

Aufruf:
    python scripts/pick_teams.py [VIDEO] [OUTPUT_JSON] [PORT]

Beispiele:
    python scripts/pick_teams.py Videos/Muenchenstein_1.mp4
    python scripts/pick_teams.py Videos/Muenchenstein_1.mp4 team_colors.json 5555
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from src.object_detection.detector import Detector
from src.tracking.team_assigner import extract_hsv_feature

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("pick_teams")

# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------
VIDEO   = sys.argv[1] if len(sys.argv) > 1 else "Videos/Muenchenstein_1.mp4"
OUTPUT  = sys.argv[2] if len(sys.argv) > 2 else "team_colors.json"
PORT    = int(sys.argv[3]) if len(sys.argv) > 3 else 5556
_MODEL  = "finetune/runs/train/weights/best.pt"
MODEL   = _MODEL if Path(_MODEL).exists() else "yolo11n.pt"

# ---------------------------------------------------------------------------
# YOLO Detection auf erstem Frame
# ---------------------------------------------------------------------------
logger.info("Lade erstes Frame aus: %s", VIDEO)
cap = cv2.VideoCapture(VIDEO)
ok, frame_orig = cap.read()
cap.release()
if not ok:
    logger.error("Konnte erstes Frame nicht lesen.")
    sys.exit(1)

logger.info("Lade YOLO-Modell …")
detector = Detector(MODEL, conf_thresholds={"player": 0.18, "goalkeeper": 0.18},
                    detect_ball=False)
dets = [d for d in detector.detect(frame_orig, frame_index=0)
        if d.class_name in ("player", "goalkeeper")]
logger.info("%d Spieler erkannt.", len(dets))

if len(dets) < 2:
    logger.error("Zu wenige Spieler erkannt (min. 2 benötigt).")
    sys.exit(1)

BBOXES = [list(map(float, d.bbox)) for d in dets]
H_IMG, W_IMG = frame_orig.shape[:2]

# Basisframe als JPEG-Base64 für den Browser
_, buf = cv2.imencode(".jpg", frame_orig, [cv2.IMWRITE_JPEG_QUALITY, 85])
FRAME_B64 = base64.b64encode(buf).decode()

# ---------------------------------------------------------------------------
# Zustand (server-seitig)
# ---------------------------------------------------------------------------
state: dict = {"team_a": None, "team_b": None}   # Index in BBOXES

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.logger.setLevel(logging.ERROR)
_shutdown_event = threading.Event()

HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Team-Farb-Picker</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; display:flex;
         flex-direction:column; align-items:center; padding:12px; }
  h2   { margin:0 0 8px; font-size:18px; }
  #hint { font-size:13px; color:#aaa; margin-bottom:10px; text-align:center; }
  #canvas-wrap { position:relative; cursor:crosshair; }
  canvas { display:block; max-width:100%; }
  #status { margin-top:10px; font-size:14px; min-height:22px; }
  #btn-save { margin-top:10px; padding:10px 32px; font-size:16px;
               background:#2a7; color:#fff; border:none; border-radius:6px;
               cursor:pointer; display:none; }
  #btn-reset { margin-top:6px; padding:6px 20px; font-size:13px;
                background:#555; color:#eee; border:none; border-radius:4px; cursor:pointer; }
  .legend { display:flex; gap:20px; margin-top:8px; font-size:13px; }
  .dot { width:14px; height:14px; border-radius:50%; display:inline-block;
         vertical-align:middle; margin-right:5px; }
</style>
</head>
<body>
<h2>Team-Farb-Picker</h2>
<div id="hint">
  Klicke auf einen Spieler → dann <b style="color:#f66">A</b> oder <b style="color:#66f">B</b>
  drücken.<br>Wenn beide gewählt sind, auf <b>Speichern</b> klicken.
</div>
<div class="legend">
  <span><span class="dot" style="background:#dc3232"></span>Team A</span>
  <span><span class="dot" style="background:#3264dc"></span>Team B</span>
  <span><span class="dot" style="background:#dcdc00; border-radius:0"></span>Auswahl</span>
</div>
<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>
<div id="status">Bitte Spieler anklicken.</div>
<button id="btn-save" onclick="save()">💾 Speichern &amp; Beenden</button>
<button id="btn-reset" onclick="reset()">↩ Zurücksetzen</button>

<script>
const IMG_W = {{ W }}, IMG_H = {{ H }};
const BBOXES = {{ BBOXES }};
const img = new Image();
img.src = "data:image/jpeg;base64,{{ FRAME }}";

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

let scale = 1;
let selectedIdx = null;
let teamA = null, teamB = null;

img.onload = () => {
  scale = Math.min(window.innerWidth * 0.96 / IMG_W, 900 / IMG_H);
  canvas.width  = IMG_W * scale;
  canvas.height = IMG_H * scale;
  draw();
};

function draw() {
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  BBOXES.forEach((bb, i) => {
    const [x1,y1,x2,y2] = bb.map(v => v * scale);
    let color = '#cccccc', lw = 2;
    if (i === teamA)           { color = '#ff3333'; lw = 4; }
    else if (i === teamB)      { color = '#3399ff'; lw = 4; }
    else if (i === selectedIdx){ color = '#ffee00'; lw = 3; }

    // Halbtransparente Füllfarbe für bessere Sichtbarkeit
    ctx.fillStyle = (i === selectedIdx) ? 'rgba(255,238,0,0.15)' :
                    (i === teamA)       ? 'rgba(255,50,50,0.15)' :
                    (i === teamB)       ? 'rgba(50,150,255,0.15)' : 'rgba(200,200,200,0.08)';
    ctx.fillRect(x1, y1, x2-x1, y2-y1);

    ctx.strokeStyle = color;
    ctx.lineWidth   = lw;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // Nummer der Box (immer sichtbar)
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = color;
    ctx.fillText('#'+i, x1+3, y2-3);

    let label = '';
    if (i === teamA)            label = 'TEAM A ✓';
    else if (i === teamB)       label = 'TEAM B ✓';
    else if (i === selectedIdx) label = '← A oder B drücken';
    if (label) {
      ctx.font = 'bold 14px sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(x1, y1 - 20, tw + 6, 18);
      ctx.fillStyle = color;
      ctx.fillText(label, x1 + 3, y1 - 5);
    }
  });
}

canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  // Viewport-relative → original image coordinates
  const cx = (e.clientX - rect.left) / scale;
  const cy = (e.clientY - rect.top)  / scale;

  // Nächste Bbox finden (kein exaktes Treffen nötig — auch Klick in der Nähe reicht)
  let best = null, bestScore = Infinity;
  BBOXES.forEach((bb, i) => {
    const [x1,y1,x2,y2] = bb;
    const bcx = (x1+x2)/2, bcy = (y1+y2)/2;
    // Distanz zum Mittelpunkt der Box
    const dist = Math.hypot(cx - bcx, cy - bcy);
    // Bonus wenn Klick innerhalb der Box liegt
    const inside = (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) ? 0 : 50;
    const score = dist + inside;
    if (score < bestScore) { bestScore = score; best = i; }
  });
  selectedIdx = best;
  updateStatus();
  draw();
});

document.addEventListener('keydown', e => {
  if (selectedIdx === null) return;
  if (e.key === 'a' || e.key === 'A') {
    teamA = selectedIdx; selectedIdx = null; updateStatus(); draw();
    fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'},
                   body: JSON.stringify({team:'a', idx: teamA})});
  } else if (e.key === 'b' || e.key === 'B') {
    teamB = selectedIdx; selectedIdx = null; updateStatus(); draw();
    fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'},
                   body: JSON.stringify({team:'b', idx: teamB})});
  }
});

function updateStatus() {
  const s = document.getElementById('status');
  const parts = [];
  if (teamA === null) parts.push('Team A: <span style="color:#f88">nicht gewählt</span>');
  else                parts.push('Team A: <span style="color:#4e4">✓ gesetzt</span>');
  if (teamB === null) parts.push('Team B: <span style="color:#88f">nicht gewählt</span>');
  else                parts.push('Team B: <span style="color:#4e4">✓ gesetzt</span>');
  if (selectedIdx !== null) parts.push('<b style="color:#dd0">Box gewählt — A oder B drücken</b>');
  s.innerHTML = parts.join(' &nbsp;|&nbsp; ');
  document.getElementById('btn-save').style.display =
    (teamA !== null && teamB !== null) ? 'inline-block' : 'none';
}

function save() {
  fetch('/save', {method:'POST'}).then(r => r.json()).then(d => {
    document.getElementById('status').innerHTML =
      '<span style="color:#4e4">✅ Gespeichert! Du kannst das Fenster schließen.</span>';
    document.getElementById('btn-save').disabled = true;
  });
}

function reset() {
  teamA = teamB = selectedIdx = null;
  fetch('/reset', {method:'POST'});
  updateStatus(); draw();
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        HTML,
        W=W_IMG, H=H_IMG,
        BBOXES=json.dumps(BBOXES),
        FRAME=FRAME_B64,
    )

@app.route("/set", methods=["POST"])
def set_team():
    data = request.get_json()
    team, idx = data["team"], int(data["idx"])
    state["team_a" if team == "a" else "team_b"] = idx
    return jsonify(ok=True)

@app.route("/reset", methods=["POST"])
def reset():
    state["team_a"] = state["team_b"] = None
    return jsonify(ok=True)

@app.route("/save", methods=["POST"])
def save():
    ia, ib = state["team_a"], state["team_b"]
    if ia is None or ib is None:
        return jsonify(ok=False, error="Nicht beide Teams gesetzt")

    feat_a = extract_hsv_feature(frame_orig, BBOXES[ia])
    feat_b = extract_hsv_feature(frame_orig, BBOXES[ib])
    if feat_a is None or feat_b is None:
        return jsonify(ok=False, error="HSV-Feature konnte nicht extrahiert werden")

    out = Path(OUTPUT)
    out.write_text(json.dumps({
        "team_a_hsv": feat_a.tolist(),
        "team_b_hsv": feat_b.tolist(),
        "bbox_a": BBOXES[ia],
        "bbox_b": BBOXES[ib],
    }, indent=2))
    logger.info("Gespeichert: %s", out.resolve())
    logger.info("  Team A HSV: H=%.0f S=%.0f V=%.0f", *feat_a)
    logger.info("  Team B HSV: H=%.0f S=%.0f V=%.0f", *feat_b)

    # Server nach kurzer Pause herunterfahren
    def _stop():
        import time; time.sleep(1); _shutdown_event.set()
    threading.Thread(target=_stop, daemon=True).start()
    return jsonify(ok=True, path=str(out.resolve()))

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
def main():
    logger.info("")
    logger.info("=" * 55)
    logger.info("  Team-Farb-Picker läuft auf:")
    logger.info("  http://localhost:%d", PORT)
    logger.info("  Öffne diese URL im Browser!")
    logger.info("=" * 55)
    logger.info("")

    server = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    server.start()
    _shutdown_event.wait()
    logger.info("Team-Farb-Picker beendet.")


if __name__ == "__main__":
    main()
