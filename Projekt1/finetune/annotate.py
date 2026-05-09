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
  <button id="btnLabels"  onclick="toggleLabels()"  title="Labels ein/ausblenden (L)">Labels: AN</button>
  <button id="btnSharpen" onclick="toggleSharpen()" title="Schärfung ein/ausblenden (K)">Schärfe: AUS</button>
  <button onclick="resetView();draw()" title="Zoom zurücksetzen (R)" style="color:#aaf">Zoom: Reset (R)</button>
  <span id="zoomHint" style="color:#aaa;font-size:12px;align-self:center">🖱 Rad=Zoom | Leertaste+Ziehen=Pan</span>
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
let showLabels = true, sharpen = false;

// ── Zoom / Pan ────────────────────────────────────────────────────────────
let zoom = 1.0, panX = 0, panY = 0;
let spaceDown = false, panDragStart = null, panAtStart = null;

function resetView(){ zoom=1.0; panX=0; panY=0; }

// Maus-Canvas → Bild-Display-Koordinaten (vor Zoom/Pan)
function canvasToDisplay(cx,cy){ return [(cx-panX)/zoom, (cy-panY)/zoom]; }
function displayToNorm(dx,dy){ return [dx/canvas.width, dy/canvas.height]; }
function mouseToNorm(mx,my){ const[dx,dy]=canvasToDisplay(mx,my); return displayToNorm(dx,dy); }
function normToDisplay(nx,ny){ return [nx*canvas.width, ny*canvas.height]; }
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
  imgIdx=idx; selected=-1; dirty=false; resetView();
  const name=images[idx];
  fetch('/labels/'+encodeURIComponent(name)).then(r=>r.json()).then(data=>{
    boxes=data.boxes;
    const img=new Image();
    img.onload=()=>{
      curImg=img;
      const maxW=window.innerWidth-20, maxH=window.innerHeight-100;
      const s=Math.min(maxW/img.width, maxH/img.height, 1.0);
      canvas.width=Math.round(img.width*s);
      canvas.height=Math.round(img.height*s);
      draw();
      setStatus(`[${idx+1}/${images.length}]  ${name}  |  Boxen: ${boxes.length}`);
    };
    img.src='/image/'+encodeURIComponent(name)+(sharpen?'?sharpen=1':'');
  });
}

function draw(){
  if(!curImg) return;
  ctx.save();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.setTransform(zoom,0,0,zoom,panX,panY);
  ctx.drawImage(curImg,0,0,canvas.width,canvas.height);

  boxes.forEach((b,i)=>{
    const[x1,y1]=normToDisplay(b.cx-b.bw/2, b.cy-b.bh/2);
    const bw=b.bw*canvas.width, bh=b.bh*canvas.height;
    const color=i===selected?'#00ddff':CLS_COLORS[b.cls%4];
    ctx.strokeStyle=color; ctx.lineWidth=(i===selected?3:2)/zoom;
    ctx.strokeRect(x1,y1,bw,bh);
    if(showLabels){
      const lbl=CLS_NAMES[b.cls]||String(b.cls);
      const fs=Math.max(10,12/zoom);
      ctx.font=`bold ${fs}px monospace`;
      const tw=ctx.measureText(lbl).width, lh=fs+4;
      ctx.fillStyle=color; ctx.fillRect(x1,y1-lh,tw+6,lh);
      ctx.fillStyle='#000'; ctx.fillText(lbl,x1+3,y1-3);
    }
  });

  if(dragStart&&dragCur&&!spaceDown){
    const[sx,sy]=canvasToDisplay(...dragStart);
    const[ex,ey]=canvasToDisplay(...dragCur);
    if(Math.abs(ex-sx)>2||Math.abs(ey-sy)>2){
      ctx.strokeStyle=CLS_COLORS[curCls]; ctx.lineWidth=2/zoom;
      ctx.setLineDash([6/zoom,3/zoom]);
      ctx.strokeRect(Math.min(sx,ex),Math.min(sy,ey),Math.abs(ex-sx),Math.abs(ey-sy));
      ctx.setLineDash([]);
    }
  }
  ctx.restore();
  setStatus(`[${imgIdx+1}/${images.length}]  ${images[imgIdx]}${dirty?' *':''}  |  Boxen: ${boxes.length}${selected>=0?' | Sel:'+selected:''}  |  Zoom: ${Math.round(zoom*100)}%`);
}

// ── Maus-Events ───────────────────────────────────────────────────────────
canvas.addEventListener('contextmenu', e=>e.preventDefault());

canvas.addEventListener('wheel', e=>{
  e.preventDefault();
  const factor=e.deltaY<0?1.15:1/1.15;
  const newZoom=Math.max(0.5,Math.min(12.0,zoom*factor));
  panX=e.offsetX-(e.offsetX-panX)*newZoom/zoom;
  panY=e.offsetY-(e.offsetY-panY)*newZoom/zoom;
  zoom=newZoom; draw();
},{passive:false});

canvas.addEventListener('mousedown', e=>{
  if(spaceDown){
    panDragStart=[e.offsetX,e.offsetY]; panAtStart=[panX,panY];
    canvas.style.cursor='grabbing'; return;
  }
  if(e.button===2){
    const[nx,ny]=mouseToNorm(e.offsetX,e.offsetY);
    const hit=hitTest(nx,ny);
    if(hit>=0){boxes.splice(hit,1);selected=-1;dirty=true;draw();}
    return;
  }
  dragStart=[e.offsetX,e.offsetY]; dragCur=[e.offsetX,e.offsetY];
});
canvas.addEventListener('mousemove', e=>{
  if(panDragStart){
    panX=panAtStart[0]+(e.offsetX-panDragStart[0]);
    panY=panAtStart[1]+(e.offsetY-panDragStart[1]);
    draw(); return;
  }
  if(dragStart){dragCur=[e.offsetX,e.offsetY];draw();}
});
canvas.addEventListener('mouseup', e=>{
  if(panDragStart){
    panDragStart=null; panAtStart=null;
    canvas.style.cursor=spaceDown?'grab':'crosshair'; return;
  }
  if(!dragStart) return;
  const[sx,sy]=dragStart,[ex,ey]=[e.offsetX,e.offsetY];
  dragStart=null; dragCur=null;
  const[dsx,dsy]=canvasToDisplay(sx,sy);
  const[dex,dey]=canvasToDisplay(ex,ey);
  if(Math.abs(dex-dsx)>=6&&Math.abs(dey-dsy)>=6){
    const[nx1,ny1]=displayToNorm(Math.min(dsx,dex),Math.min(dsy,dey));
    const[nx2,ny2]=displayToNorm(Math.max(dsx,dex),Math.max(dsy,dey));
    boxes.push({cls:curCls,cx:clamp((nx1+nx2)/2),cy:clamp((ny1+ny2)/2),
                bw:clamp(nx2-nx1),bh:clamp(ny2-ny1)});
    selected=boxes.length-1; dirty=true;
  } else {
    const[nx,ny]=mouseToNorm(ex,ey); selected=hitTest(nx,ny);
  }
  draw();
});

function clamp(v){return Math.max(0,Math.min(1,v));}

function hitTest(nx,ny){
  let best=-1,bestArea=Infinity;
  boxes.forEach((b,i)=>{
    if(nx>=b.cx-b.bw/2&&nx<=b.cx+b.bw/2&&ny>=b.cy-b.bh/2&&ny<=b.cy+b.bh/2){
      const a=b.bw*b.bh; if(a<bestArea){bestArea=a;best=i;}
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
function toggleSharpen(){
  sharpen=!sharpen;
  const btn=document.getElementById('btnSharpen');
  btn.textContent='Schärfe: '+(sharpen?'AN':'AUS');
  btn.style.color=sharpen?'#8f8':'#f88';
  loadImage(imgIdx);
}
function deleteSelected(){
  if(selected>=0){boxes.splice(selected,1);selected=-1;dirty=true;draw();}
}
function clearAll(){
  if(!confirm('Alle Boxen löschen?')) return;
  boxes=[];selected=-1;dirty=true;draw();
}
async function saveLabels(){
  const name=images[imgIdx];
  await fetch('/labels/'+encodeURIComponent(name),
    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({boxes})});
  dirty=false; draw();
  setStatus(`Gespeichert: ${name}  (${boxes.length} Boxen)`);
}
async function nextImg(){await saveLabels();loadImage((imgIdx+1)%images.length);}
async function prevImg(){await saveLabels();loadImage((imgIdx-1+images.length)%images.length);}

// ── Tastatur ──────────────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.code==='Space'&&!spaceDown){
    spaceDown=true; canvas.style.cursor='grab'; e.preventDefault(); return;
  }
  if(e.key==='n'||e.key==='Enter')          {e.preventDefault();nextImg();}
  else if(e.key==='p'||e.key==='Backspace') {e.preventDefault();prevImg();}
  else if(e.key==='s') saveLabels();
  else if(e.key==='d') deleteSelected();
  else if(e.key==='a') clearAll();
  else if(e.key==='r') {resetView();draw();}
  else if(e.key==='Tab'){e.preventDefault();
    if(boxes.length){selected=(selected+1)%boxes.length;draw();}}
  else if('0123'.includes(e.key)) setCls(parseInt(e.key));
  else if(e.key==='l'||e.key==='L') toggleLabels();
  else if(e.key==='k'||e.key==='K') toggleSharpen();
});
document.addEventListener('keyup', e=>{
  if(e.code==='Space'){
    spaceDown=false; panDragStart=null;
    canvas.style.cursor='crosshair';
  }
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
            name     = path[7:]
            qs       = urllib.parse.parse_qs(self.path.split("?")[1]) if "?" in self.path else {}
            sharpen  = qs.get("sharpen", ["0"])[0] == "1"
            img_path = self.img_dir / name
            if img_path.exists():
                if sharpen:
                    import cv2, numpy as np
                    img = cv2.imread(str(img_path))
                    # Unsharp Mask: geschärftes Bild = original + (original - blur)
                    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
                    sharpened = cv2.addWeighted(img, 1.8, blurred, -0.8, 0)
                    _, buf = cv2.imencode(".jpg", sharpened, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    self._send(200, "image/jpeg", buf.tobytes())
                else:
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
