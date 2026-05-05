"""UniVision2Board — Haupt-Einstiegspunkt.

Führt die komplette Pipeline in einem Schritt aus:
  1. (Optional) Team-Farben interaktiv kalibrieren
  2. Spielererkennung, Tracking und Teamzuordnung
  3. Annotiertes Video mit Taktikboard-Overlay erzeugen
  4. Ausgabevideo automatisch öffnen

Aufruf:
    python run_pipeline.py [VIDEO] [N_FRAMES] [OUTPUT] [CONF]

Beispiele:
    python run_pipeline.py
    python run_pipeline.py Videos/Muenchenstein_1.mp4
    python run_pipeline.py Videos/Muenchenstein_1.mp4 300 analyse.mp4 0.20

Hinweise:
    - Beim ersten Lauf: Team-Farb-Kalibrierung im Browser (Port 5556)
    - Danach wird team_colors.json wiederverwendet
    - Mit --reset-teams wird eine neue Kalibrierung erzwungen
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
import webbrowser
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------
VIDEO      = sys.argv[1] if len(sys.argv) > 1 else "Videos/Muenchenstein_1.mp4"
N_FRAMES   = sys.argv[2] if len(sys.argv) > 2 else "300"
OUTPUT     = sys.argv[3] if len(sys.argv) > 3 else "output_annotated.mp4"
CONF       = sys.argv[4] if len(sys.argv) > 4 else "0.20"
RESET      = "--reset-teams" in sys.argv

TEAM_FILE  = _ROOT / "team_colors.json"
PICK_PORT  = 5556
PYTHON     = sys.executable

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)

def _step(n: int, text: str) -> None:
    print(f"\n[Schritt {n}] {text}")

def _ok(text: str) -> None:
    print(f"  ✓  {text}")

def _warn(text: str) -> None:
    print(f"  ⚠  {text}")

# ---------------------------------------------------------------------------
# Schritt 1: Team-Farb-Kalibrierung (falls nötig)
# ---------------------------------------------------------------------------
_header("UniVision2Board — Pipeline starten")

print(f"  Video:   {VIDEO}")
print(f"  Frames:  {N_FRAMES}")
print(f"  Output:  {OUTPUT}")
print(f"  Konfidenz: {CONF}")

if RESET and TEAM_FILE.exists():
    TEAM_FILE.unlink()
    _warn("team_colors.json gelöscht — neue Kalibrierung wird durchgeführt.")

if not TEAM_FILE.exists():
    _step(1, "Team-Farb-Kalibrierung (einmalig)")
    print()
    print("  Öffne Browser auf http://localhost:{} …".format(PICK_PORT))
    print("  Bedienung:")
    print("    1. Klicke auf einen Spieler von Team A")
    print("    2. Drücke Taste  A")
    print("    3. Klicke auf einen Spieler von Team B")
    print("    4. Drücke Taste  B")
    print("    5. Klicke  Speichern & Beenden")
    print()

    # pick_teams.py als Subprocess starten
    pick_proc = subprocess.Popen(
        [PYTHON, str(_ROOT / "scripts" / "pick_teams.py"), VIDEO,
         str(TEAM_FILE), str(PICK_PORT)],
        cwd=str(_ROOT),
    )

    # Browser nach kurzer Pause öffnen
    def _open_browser():
        time.sleep(3)
        webbrowser.open(f"http://localhost:{PICK_PORT}")
    threading.Thread(target=_open_browser, daemon=True).start()

    # Warten bis team_colors.json erscheint (= Picker hat gespeichert)
    print("  Warte auf Kalibrierung …", end="", flush=True)
    while not TEAM_FILE.exists():
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    pick_proc.terminate()
    _ok(f"Team-Farben gespeichert: {TEAM_FILE}")
else:
    _step(1, "Team-Farb-Kalibrierung")
    _ok(f"Vorhandene Kalibrierung wird verwendet: {TEAM_FILE}")
    print("    (Mit --reset-teams neu kalibrieren)")

# ---------------------------------------------------------------------------
# Schritt 2: Pipeline ausführen
# ---------------------------------------------------------------------------
_step(2, f"Pipeline starten ({N_FRAMES} Frames) …")
print()

pipeline_args = [
    PYTHON,
    str(_ROOT / "scripts" / "visualize_pipeline.py"),
    VIDEO, N_FRAMES, OUTPUT, CONF,
]

start = time.time()
result = subprocess.run(pipeline_args, cwd=str(_ROOT))
elapsed = time.time() - start

if result.returncode != 0:
    print(f"\n  ✗  Pipeline fehlgeschlagen (Exit-Code {result.returncode})")
    sys.exit(result.returncode)

# ---------------------------------------------------------------------------
# Schritt 3: Ausgabe öffnen
# ---------------------------------------------------------------------------
_step(3, "Fertig!")
out_path = _ROOT / OUTPUT
print(f"\n  Ausgabevideo: {out_path}")
print(f"  Dauer:        {elapsed:.0f}s ({float(N_FRAMES)/elapsed:.1f} Frames/s)")
print()

try:
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(out_path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(out_path)])
    else:
        os.startfile(str(out_path))
    _ok("Video wird geöffnet …")
except Exception as e:
    _warn(f"Video konnte nicht automatisch geöffnet werden: {e}")
    print(f"  Manuell öffnen mit:  xdg-open {out_path}")
