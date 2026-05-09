"""Phase 6: Interaktives Taktikboard (Tkinter).

Zeigt das Unihockey-Taktikboard mit Spieler-Symbolen, Laufwegen und
taktischen Pfeilen.  Der Trainer kann:
  - Zeichenmodus wählen (Pass, Schuss, Laufweg) und Pfeile einzeichnen
  - Spieler per Klick auswählen und Team-Zuordnung korrigieren
  - Auf die nächste/vorherige Frame-Ansicht navigieren

Aufruf:
    Nicht direkt – wird von tactic_board_app.py instanziiert und gestartet.

Internes Koordinatensystem:
    Canvas-Pixel  ←→  Feldkoordinaten (Meter) via BoardRenderer.board_to_canvas()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Callable

from PIL import Image, ImageTk

from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol, Arrow
from src.tracking.team_assigner import TEAM_A, TEAM_B
from src.utils.exporter import export_png, export_pdf, preview_temp

# Zeichenmodi
MODE_SELECT = "select"
MODE_PASS   = "pass"
MODE_SHOT   = "shot"
MODE_RUN    = "run"


class TacticBoardApp:
    """Hauptfenster des Taktikboards.

    Args:
        states:          Geordnete Liste von BoardState-Objekten (ein State pro Frame).
        renderer:        BoardRenderer-Instanz (enthält Hintergrundbild + Koordinaten-Mapping).
        on_team_override: Callback: (track_id, new_team) → None (für TeamAssigner).
    """

    def __init__(
        self,
        states: list[BoardState],
        renderer: BoardRenderer,
        on_team_override: Callable[[int, int], None] | None = None,
    ) -> None:
        self._states = states
        self._renderer = renderer
        self._on_team_override = on_team_override
        self._current_idx = 0
        self._mode = MODE_SELECT
        self._draw_start: tuple[int, int] | None = None
        self._selected_player: PlayerSymbol | None = None
        self._extra_arrows: list[Arrow] = []   # manuell gezeichnete Pfeile

        self._root = tk.Tk()
        self._root.title("UniVision2Board – Taktikboard")
        self._root.resizable(False, False)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI aufbauen
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Toolbar oben
        toolbar = ttk.Frame(self._root, padding=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="Modus:").pack(side=tk.LEFT, padx=(0, 4))
        self._mode_var = tk.StringVar(value=MODE_SELECT)
        for label, mode in [("Auswahl", MODE_SELECT), ("Pass ▶", MODE_PASS),
                             ("Schuss ▶", MODE_SHOT), ("Laufweg ▶", MODE_RUN)]:
            rb = ttk.Radiobutton(toolbar, text=label, variable=self._mode_var,
                                 value=mode, command=self._on_mode_change)
            rb.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        self._btn_clear = ttk.Button(toolbar, text="Pfeile löschen",
                                     command=self._clear_arrows)
        self._btn_clear.pack(side=tk.LEFT, padx=2)

        # Frame-Navigation rechts in der Toolbar
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)
        ttk.Button(toolbar, text="◀ Zurück", command=self._prev_frame).pack(side=tk.LEFT)
        self._frame_label = ttk.Label(toolbar, text="Frame 1 / 1", width=14)
        self._frame_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Vor ▶", command=self._next_frame).pack(side=tk.LEFT)

        # Export-Buttons
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)
        ttk.Button(toolbar, text="💾 PNG",     command=self._export_png).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📄 PDF",     command=self._export_pdf).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="👁 Vorschau", command=self._preview).pack(side=tk.LEFT, padx=2)

        # Legende rechts
        legend = ttk.Frame(self._root, padding=6)
        legend.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_legend(legend)

        # Canvas
        cw, ch = self._renderer.canvas_w, self._renderer.canvas_h
        self._canvas = tk.Canvas(self._root, width=cw, height=ch,
                                 cursor="crosshair", bg="white")
        self._canvas.pack(side=tk.LEFT)
        self._canvas.bind("<Button-1>",        self._on_click)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>",          self._on_motion)

        # Statuszeile
        self._status_var = tk.StringVar(value="Bereit.")
        ttk.Label(self._root, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

        # Hotkeys
        self._root.bind("<Left>",  lambda _: self._prev_frame())
        self._root.bind("<Right>", lambda _: self._next_frame())
        self._root.bind("p", lambda _: self._set_mode(MODE_PASS))
        self._root.bind("s", lambda _: self._set_mode(MODE_SHOT))
        self._root.bind("r", lambda _: self._set_mode(MODE_RUN))
        self._root.bind("<Escape>", lambda _: self._set_mode(MODE_SELECT))

        self._refresh()

    def _build_legend(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Legende", font=("TkDefaultFont", 9, "bold")).pack()
        items = [
            ("● Team A",    "#DC2828"),
            ("● Team B",    "#1E5ADC"),
            ("■ Torwart",   "#B43CDC"),
            ("● Ball",      "#F0C800"),
            ("─▶ Pass",     "#14B414"),
            ("─▶ Schuss",   "#DC3C14"),
            ("╌▶ Laufweg",  "#3CA0DC"),
        ]
        for text, color in items:
            ttk.Label(parent, text=text, foreground=color).pack(anchor=tk.W)
        ttk.Separator(parent).pack(fill=tk.X, pady=4)
        ttk.Label(parent, text="Hotkeys", font=("TkDefaultFont", 9, "bold")).pack()
        for line in ["P = Pass", "S = Schuss", "R = Laufweg", "ESC = Auswahl",
                     "◀ ▶ = Frames"]:
            ttk.Label(parent, text=line, foreground="#444").pack(anchor=tk.W)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        state = self._merged_state()
        img = self._renderer.render_rgb(state)
        self._tk_img = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        total = max(len(self._states), 1)
        self._frame_label.config(text=f"Frame {self._current_idx + 1} / {total}")

    def _merged_state(self) -> BoardState:
        """Aktueller State + manuell gezeichnete Extra-Pfeile."""
        if not self._states:
            return BoardState()
        base = self._states[self._current_idx]
        return BoardState(
            players=base.players,
            arrows=base.arrows + self._extra_arrows,
            frame_index=base.frame_index,
        )

    # ------------------------------------------------------------------
    # Event-Handler
    # ------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        self._mode = self._mode_var.get()
        cursor = "crosshair" if self._mode != MODE_SELECT else "arrow"
        self._canvas.config(cursor=cursor)
        self._status_var.set(f"Modus: {self._mode}")

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._mode_var.set(mode)
        self._on_mode_change()

    def _on_click(self, event: tk.Event) -> None:
        if self._mode == MODE_SELECT:
            self._try_select_player(event.x, event.y)
        else:
            self._draw_start = (event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self._mode != MODE_SELECT and self._draw_start:
            sx, sy = self._draw_start
            if abs(event.x - sx) > 5 or abs(event.y - sy) > 5:
                bx0, by0 = self._renderer.canvas_to_board(sx, sy)
                bx1, by1 = self._renderer.canvas_to_board(event.x, event.y)
                self._extra_arrows.append(
                    Arrow(bx0, by0, bx1, by1, kind=self._mode)
                )
                self._refresh()
            self._draw_start = None

    def _on_motion(self, event: tk.Event) -> None:
        bx, by = self._renderer.canvas_to_board(event.x, event.y)
        self._status_var.set(f"Modus: {self._mode}  |  Feld: ({bx:.1f} m, {by:.1f} m)")

    def _try_select_player(self, cx: int, cy: int) -> None:
        """Wählt den nächstliegenden Spieler aus und öffnet Korrektur-Dialog."""
        if not self._states:
            return
        state = self._states[self._current_idx]
        best: PlayerSymbol | None = None
        best_dist = 40   # Pixel-Schwelle für Klick-Treffer

        for p in state.players:
            px, py = self._renderer.board_to_canvas(p.board_x, p.board_y)
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = p

        if best:
            self._selected_player = best
            self._show_player_menu(best)

    def _show_player_menu(self, player: PlayerSymbol) -> None:
        """Kontextmenü für Team-Korrektur."""
        menu = tk.Menu(self._root, tearoff=0)
        team_str = "A" if player.team == TEAM_A else ("B" if player.team == TEAM_B else "?")
        menu.add_command(label=f"Spieler #{player.track_id}  (Team {team_str})",
                         state=tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="→ Team A",
                         command=lambda: self._override_team(player, TEAM_A))
        menu.add_command(label="→ Team B",
                         command=lambda: self._override_team(player, TEAM_B))
        try:
            menu.tk_popup(
                self._root.winfo_pointerx(),
                self._root.winfo_pointery(),
            )
        finally:
            menu.grab_release()

    def _override_team(self, player: PlayerSymbol, new_team: int) -> None:
        player.team = new_team
        if self._on_team_override:
            self._on_team_override(player.track_id, new_team)
        self._refresh()
        label = "A" if new_team == TEAM_A else "B"
        self._status_var.set(f"Spieler #{player.track_id} → Team {label} korrigiert.")

    def _prev_frame(self) -> None:
        if self._current_idx > 0:
            self._current_idx -= 1
            self._refresh()

    def _next_frame(self) -> None:
        if self._current_idx < len(self._states) - 1:
            self._current_idx += 1
            self._refresh()

    def _clear_arrows(self) -> None:
        self._extra_arrows.clear()
        self._refresh()

    # ------------------------------------------------------------------
    # Export (Phase 7)
    # ------------------------------------------------------------------

    def _export_png(self) -> None:
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG-Bild", "*.png")],
            initialfile="taktikboard.png",
            title="Board als PNG speichern",
        )
        if path:
            saved = export_png(self.get_current_image(), path)
            self._status_var.set(f"PNG gespeichert: {saved}")

    def _export_pdf(self) -> None:
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF-Dokument", "*.pdf")],
            initialfile="taktikboard.pdf",
            title="Board als PDF speichern",
        )
        if path:
            saved = export_pdf(self.get_current_image(), path)
            self._status_var.set(f"PDF gespeichert: {saved}")

    def _preview(self) -> None:
        """Temporäre Vorschau – öffnet Systembetrachter, keine persistente Datei im Projekt."""
        preview_temp(self.get_current_image(), open_viewer=True)
        self._status_var.set("Vorschau geöffnet (temporäre Datei, kein Projektexport).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Startet die Tkinter-Mainloop (blockierend)."""
        self._root.mainloop()

    def get_current_image(self) -> Image.Image:
        """Gibt das aktuell angezeigte Board-Bild zurück (für Export)."""
        return self._renderer.render_rgb(self._merged_state())

    def get_extra_arrows(self) -> list[Arrow]:
        """Gibt alle manuell gezeichneten Pfeile zurück."""
        return list(self._extra_arrows)
