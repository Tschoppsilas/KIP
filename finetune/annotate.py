"""Web-basiertes Annotations-Tool für YOLO-Datensätze.

Startet einen lokalen Webserver und öffnet das Tool im Browser.
Keine GUI-Bibliothek nötig – funktioniert auf jedem System.

Aufruf:
  python finetune/annotate.py                 # train-Split, Port 8080
  python finetune/annotate.py --split val
  python finetune/annotate.py --port 8888
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR  = PROJECT_ROOT / "finetune" / "dataset"

CLASS_NAMES  = ["player", "goalkeeper", "ball", "referee"]
CLASS_COLORS = ["#22cc44", "#22ccff", "#ff8822", "#cc22ff"]


# ── YOLO ↔ Normalisiert ───────────────────────────────────────────────────────

def load_labels(path: Path) -> list[dict]:
    """Liest YOLO-Labels → [{cls, cx, cy, bw, bh}, ...]"""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            boxes.append({
                "cls": int(parts[0]),
                "cx": float(parts[1]), "cy": float(parts[2]),
                "bw": float(parts[3]), "bh": float(parts[4]),
            })
        except ValueError:
            pass
    return boxes


def save_labels(path: Path, boxes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['bw']:.6f} {b['bh']:.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ── HTML / JS ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Annotate</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a1a; color: #ddd; font-family: monospace; display: flex;
       flex-direction: column; height: 100vh; overflow: hidden; }
#toolbar { background: #111; padding: 6px 12px; display: flex; align-items: center;
           gap: 12px; flex-wrap: wrap; flex-shrink: 0; border-bottom: 1px solid #333; }
#toolbar button { background: #2a2a2a; color: #ddd; border: 1px solid #444;
                  padding: 4px 10px; cursor: pointer; border-radius: 3px; font: inherit; }
#toolbar button:hover { background: #3a3a3a; }
#toolbar button.active { background: #0055aa; border-color: #0077ff; }
.cls-btn { padding: 4px 10px !important; font-weight: bold; }
#status { background: #111; padding: 4px 12px; font-size: 12px; color: #aaa;
          flex-shrink: 0; border-top: 1px solid #333; }
#canvas-wrap { flex: 1; overflow: auto; display: flex;
               justify-content: center; align-items: flex-start; padding: 8px; }
canvas { cursor: crosshair; display: block; }
#shortcut-hint { color: #555; font-size: 11px; margin-left: auto; }
</style>
</head>
<body>
<div id="toolbar">
  <button onclick="prevImg()">&#9664; Zurück (P)</button>
  <button onclick="nextImg()">Weiter (N) &#9654;</button>
  <button onclick="saveLabels()" style="color:#8f8;">Speichern (S)</button>
  <button onclick="deleteSelected()" style="color:#f88;">Löschen (D)</button>
  <button onclick="clearAll()" style="color:#f44;">Alle löschen (A)</button>
  <button id="btnLabels" onclick="toggleLabels()" title="Labels ein/ausblenden (L)">Labels: AN</button>
  <span style="color:#555;">|</span>
  KLASSE:
  <button class="cls-btn" id="cls0" onclick="setCls(0)">0 player</button>
  <button class="cls-btn" id="cls1" onclick="setCls(1)">1 goalkeeper</button>
  <button class="cls-btn" id="cls2" onclick="setCls(2)">2 ball</button>
  <button class="cls-btn" id="cls3" onclick="setCls(3)">3 referee</button>
  <span id="shortcut-hint">Maus: Ziehen=Box  Klick=auswählen  Rechtsklick=löschen</span>
</div>
<div id="canvas-wrap"><canvas id="c"></canvas></div>
<div id="status">Lädt…</div>

<script>
const CLS_COLORS = ["#22cc44","#22ccff","#ff8822","#cc22ff"];
const CLS_NAMES  = ["player","goalkeeper","ball","referee"];

let images = [], imgIdx = 0, curCls = 0, curImg = null;
let boxes = [], selected = -1, dirty = false;
let dragStart = null, dragCur = null;
let showLabels = true;
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

// ── Init ──────────────────────────────────────────────────────────────────
fetch('/list').then(r=>r.json()).then(data=>{
  images = data.images;
  loadImage(0);
  updateClsButtons();
});

function setStatus(msg){ document.getElementById('status').textContent = msg; }

function updateClsButtons(){
  CLS_NAMES.forEach((_,i)=>{
    const b = document.getElementById('cls'+i);
    b.style.borderColor = i===curCls ? CLS_COLORS[i] : '#444';
    b.style.color       = CLS_COLORS[i];
    b.style.background  = i===curCls ? '#1a2a1a' : '#2a2a2a';
  });
}

function setCls(c){ curCls=c; updateClsButtons();
  if(selected>=0){ boxes[selected].cls=c; dirty=true; draw(); } }

// ── Bilder laden ──────────────────────────────────────────────────────────
function loadImage(idx){
  if(idx<0||idx>=images.length) return;
  imgIdx = idx; selected=-1; dirty=false;
  const name = images[idx];
  fetch('/labels/'+encodeURIComponent(name)).then(r=>r.json()).then(data=>{
    boxes = data.boxes;
    const img = new Image();
    img.onload = ()=>{
      curImg = img;
      // Bild auf max. Fenstergrösse skalieren
      const maxW = window.innerWidth  - 20;
      const maxH = window.innerHeight - 100;
      const s    = Math.min(maxW/img.width, maxH/img.height, 1.0);
      canvas.width  = Math.round(img.width  * s);
      canvas.height = Math.round(img.height * s);
      canvas.dataset.scale = s;
      draw();
      setStatus(`[${idx+1}/${images.length}]  ${name}  |  Boxen: ${boxes.length}`);
    };
    img.src = '/image/'+encodeURIComponent(name);
  });
}

function draw(){
  if(!curImg) return;
  const s = parseFloat(canvas.dataset.scale||1);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(curImg,0,0,canvas.width,canvas.height);

  // Boxen zeichnen
  boxes.forEach((b,i)=>{
    const x1 = (b.cx-b.bw/2)*canvas.width;
    const y1 = (b.cy-b.bh/2)*canvas.height;
    const bw = b.bw*canvas.width, bh=b.bh*canvas.height;
    ctx.strokeStyle = i===selected ? '#00ddff' : CLS_COLORS[b.cls%4];
    ctx.lineWidth   = i===selected ? 3 : 2;
    ctx.strokeRect(x1,y1,bw,bh);
    // Label
    if(showLabels){
      const lbl = CLS_NAMES[b.cls]||String(b.cls);
      ctx.font='bold 12px monospace';
      const tw=ctx.measureText(lbl).width;
      ctx.fillStyle=ctx.strokeStyle;
      ctx.fillRect(x1,y1-17,tw+6,17);
      ctx.fillStyle='#000';
      ctx.fillText(lbl,x1+3,y1-4);
    }
  });

  // Laufende Zeichnung
  if(dragStart && dragCur){
    const [x1,y1]=dragStart, [x2,y2]=dragCur;
    if(Math.abs(x2-x1)>4||Math.abs(y2-y1)>4){
      ctx.strokeStyle=CLS_COLORS[curCls]; ctx.lineWidth=2;
      ctx.setLineDash([5,3]);
      ctx.strokeRect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1));
      ctx.setLineDash([]);
    }
  }
  setStatus(`[${imgIdx+1}/${images.length}]  ${images[imgIdx]}${dirty?' *':''}  |  Boxen: ${boxes.length}${selected>=0?'  |  Ausgewählt: '+selected:''}`);
}

// ── Maus-Events ───────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e=>{
  if(e.button===2){
    // Rechtsklick → löschen
    const hit=hitTest(e.offsetX, e.offsetY);
    if(hit>=0){ boxes.splice(hit,1); selected=-1; dirty=true; draw(); }
    return;
  }
  dragStart=[e.offsetX,e.offsetY]; dragCur=[e.offsetX,e.offsetY];
});
canvas.addEventListener('mousemove', e=>{
  if(dragStart){ dragCur=[e.offsetX,e.offsetY]; draw(); }
});
canvas.addEventListener('mouseup', e=>{
  if(!dragStart) return;
  const [sx,sy]=dragStart, ex=e.offsetX, ey=e.offsetY;
  const dx=Math.abs(ex-sx), dy=Math.abs(ey-sy);
  dragStart=null; dragCur=null;
  if(dx>=8 && dy>=8){
    // Neue Box
    const x1=Math.min(sx,ex), y1=Math.min(sy,ey);
    const x2=Math.max(sx,ex), y2=Math.max(sy,ey);
    const cx=(x1+x2)/2/canvas.width,  cy=(y1+y2)/2/canvas.height;
    const bw=(x2-x1)/canvas.width,    bh=(y2-y1)/canvas.height;
    boxes.push({cls:curCls, cx:clamp(cx), cy:clamp(cy),
                bw:clamp(bw), bh:clamp(bh)});
    selected=boxes.length-1; dirty=true;
  } else {
    selected=hitTest(sx,sy);
  }
  draw();
});
canvas.addEventListener('contextmenu', e=>e.preventDefault());

function clamp(v){ return Math.max(0,Math.min(1,v)); }

function hitTest(mx,my){
  let best=-1, bestArea=Infinity;
  boxes.forEach((b,i)=>{
    const x1=(b.cx-b.bw/2)*canvas.width,  y1=(b.cy-b.bh/2)*canvas.height;
    const x2=(b.cx+b.bw/2)*canvas.width,  y2=(b.cy+b.bh/2)*canvas.height;
    if(mx>=x1&&mx<=x2&&my>=y1&&my<=y2){
      const a=(x2-x1)*(y2-y1);
      if(a<bestArea){ bestArea=a; best=i; }
    }
  });
  return best;
}

// ── Aktionen ──────────────────────────────────────────────────────────────
function toggleLabels(){
  showLabels=!showLabels;
  const btn=document.getElementById('btnLabels');
  btn.textContent='Labels: '+(showLabels?'AN':'AUS');
  btn.style.color=showLabels?'#8f8':'#f88';
  draw();
}

function deleteSelected(){
  if(selected>=0){ boxes.splice(selected,1); selected=-1; dirty=true; draw(); }}

function clearAll(){ if(!confirm('Alle Boxen löschen?')) return;
  boxes=[]; selected=-1; dirty=true; draw(); }

async function saveLabels(){
  const name=images[imgIdx];
  await fetch('/labels/'+encodeURIComponent(name),
    {method:'POST', headers:{'Content-Type':'application/json'},
     body:JSON.stringify({boxes})});
  dirty=false; draw();
  setStatus(`Gespeichert: ${name}  (${boxes.length} Boxen)`);
}

async function nextImg(){
  await saveLabels();
  loadImage((imgIdx+1)%images.length);
}
async function prevImg(){
  await saveLabels();
  loadImage((imgIdx-1+images.length)%images.length);
}

// ── Tastatur ──────────────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='n'||e.key===' '||e.key==='Enter') { e.preventDefault(); nextImg(); }
  else if(e.key==='p'||e.key==='Backspace')      { e.preventDefault(); prevImg(); }
  else if(e.key==='s')  saveLabels();
  else if(e.key==='d')  deleteSelected();
  else if(e.key==='a')  clearAll();
  else if(e.key==='Tab'){ e.preventDefault();
    if(boxes.length){ selected=(selected+1)%boxes.length; draw(); }}
  else if('0123'.includes(e.key)) setCls(parseInt(e.key));
  else if(e.key==='l'||e.key==='L') toggleLabels();
});
</script>
</body></html>
"""


# ── HTTP-Handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    images:   list[Path] = []
    img_dir:  Path       = Path()
    lbl_dir:  Path       = Path()

    def log_message(self, fmt, *args):
        pass   # Konsolenausgabe unterdrücken

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urllib.parse.unquote(self.path.split("?")[0])

        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML.encode())

        elif path == "/list":
            data = json.dumps({"images": [p.name for p in self.images]}).encode()
            self._send(200, "application/json", data)

        elif path.startswith("/image/"):
            name = path[7:]
            img_path = self.img_dir / name
            if img_path.exists():
                self._send(200, "image/jpeg", img_path.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        elif path.startswith("/labels/"):
            name    = path[8:]
            stem    = Path(name).stem
            lbl_path = self.lbl_dir / (stem + ".txt")
            boxes   = load_labels(lbl_path)
            self._send(200, "application/json", json.dumps({"boxes": boxes}).encode())

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        path = urllib.parse.unquote(self.path)
        if path.startswith("/labels/"):
            name    = path[8:]
            stem    = Path(name).stem
            lbl_path = self.lbl_dir / (stem + ".txt")
            length  = int(self.headers.get("Content-Length", 0))
            body    = json.loads(self.rfile.read(length))
            save_labels(lbl_path, body["boxes"])
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain", b"not found")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def run(split: str, port: int, batch: int, batch_size: int) -> None:
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not all_images:
        print(f"\nKeine Bilder unter: {img_dir}")
        print("Erst Frames extrahieren:  python finetune/prepare_dataset.py")
        sys.exit(1)

    # Batch-Auswahl
    n_total   = len(all_images)
    n_batches = (n_total + batch_size - 1) // batch_size
    if batch < 0 or batch >= n_batches:
        print(f"\nUngültiger Batch {batch}. Verfügbar: 0 – {n_batches-1}")
        sys.exit(1)
    start  = batch * batch_size
    images = all_images[start : start + batch_size]

    Handler.images  = images
    Handler.img_dir = img_dir
    Handler.lbl_dir = lbl_dir

    url = f"http://localhost:{port}"
    print(f"\nBatch {batch+1}/{n_batches}  |  Bilder {start+1}–{start+len(images)} von {n_total}  |  Split: {split}")
    print(f"Nächster Batch:  python finetune/annotate.py --split {split} --batch {batch+1}" if batch+1 < n_batches else "Das war der letzte Batch!")
    print(f"\nAnnotierungs-Tool läuft unter:  {url}")
    print("Öffne die URL im Browser (Firefox).")
    print("Zum Beenden: Strg+C\n")

    # Browser automatisch öffnen
    def _open():
        import time; time.sleep(0.8)
        subprocess.Popen(
            ["firefox", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    threading.Thread(target=_open, daemon=True).start()

    server = HTTPServer(("localhost", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer beendet.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-Annotations-Tool für YOLO")
    parser.add_argument("--split",      default="train", choices=["train", "val"])
    parser.add_argument("--port",       type=int, default=8080)
    parser.add_argument("--batch",      type=int, default=0,
                        help="Batch-Nummer (0-basiert, default: 0)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Bilder pro Batch (default: 20)")
    # Übersicht aller Batches anzeigen
    parser.add_argument("--list", action="store_true",
                        help="Alle verfügbaren Batches auflisten und beenden")
    args = parser.parse_args()

    if args.list:
        img_dir = DATASET_DIR / "images" / args.split
        all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n = len(all_imgs); bs = args.batch_size
        n_batches = (n + bs - 1) // bs
        print(f"\n{n} Bilder im '{args.split}'-Split  →  {n_batches} Batches à {bs} Bilder\n")
        for b in range(n_batches):
            s = b * bs
            print(f"  Batch {b:3d}:  Bilder {s+1:4d}–{min(s+bs,n):4d}"
                  f"   →  python finetune/annotate.py --split {args.split} --batch {b}")
        print()
        sys.exit(0)

    run(args.split, args.port, args.batch, args.batch_size)


if __name__ == "__main__":
    main()
