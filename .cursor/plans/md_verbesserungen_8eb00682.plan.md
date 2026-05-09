---
name: MD Verbesserungen
overview: Gezielte Optimierungsvorschläge für die drei Planungs-Markdowns auf Basis des Exposés, ohne Dateien zu ändern.
todos:
  - id: spec-schärfen
    content: "`spec.md` in klare Architekturblöcke mit sauberer MVP/Stretch-Trennung überführen."
    status: pending
  - id: plan-messbar
    content: "`prompt_plan.md` um Deliverables, Akzeptanzkriterien und Risiken pro Phase ergänzen."
    status: pending
  - id: todo-operationalisieren
    content: "`todo.md` in priorisierte, testbare DoD-Checkliste umformen."
    status: pending
isProject: false
---

# Verbesserungsvorschläge für `planing_mds`

## Ziel

Die drei Markdown-Dateien inhaltlich und strukturell so schärfen, dass sie das Exposé präziser abbilden, direkt umsetzbar sind und als belastbare Arbeitsgrundlage für Entwicklung und Abgabe dienen.

## Beobachtungen

- Die Kernidee aus dem Exposé ist gut übernommen (YOLO, ByteTrack, Homography, Teamzuordnung, GUI, Export).
- Es gibt mehrere Formatierungs- und Konsistenzprobleme (fehlende Leerzeichen/Umbrüche, Tippfehler, gemischte Bullet-Syntax).
- Manche Anforderungen aus dem Exposé fehlen oder sind zu wenig konkret (Rollen/Verantwortung, Datenschutz, Erfolgskriterien, TDD-Umfang).

## Konkreter Verbesserungsplan

- `spec.md` fachlich präzisieren:
  - Architektur in Input/Processing/Output strukturieren.
  - MVP vs. Stretch Goals klar trennen.
  - Nicht validierte Komponenten als „optional/später“ markieren (z. B. RL/Stable-Baselines3).
- `prompt_plan.md` ausführbar machen:
  - Jeden Abschnitt mit `Deliverable`, `Done-Kriterium`, `Risiko` ergänzen.
  - Abhängigkeiten zwischen Phasen explizit machen (z. B. Homography vor GUI-Overlay-Validierung).
  - Akzeptanztests mit messbaren Schwellen einführen (z. B. Tracking-ID-Switch-Rate, Teamzuordnungsquote).
- `todo.md` als echte Kontrollliste schärfen:
  - „Definition of Done“ je Phase ergänzen.
  - Offene Punkte in prüfbare Tasks aufteilen (ein Verb + ein Ergebnis pro Bullet).
  - Prioritätsmarkierung (`Must/Should/Could`) ergänzen.

## Qualitätskriterien für die Überarbeitung

- Einheitliche Sprache und Terminologie (z. B. immer „Taktikboard“, immer „Teamzuordnung“).
- Einheitliche Markdown-Syntax (Bullets, Überschriften, Leerzeilen, keine verschmolzenen Zeilen).
- Jeder Abschnitt beantwortet: Was wird gebaut? Woran messen wir Erfolg? Wie testen wir es?

## Dateien im Fokus

- [/home/admin/KIP/planing_mds/spec.md](/home/admin/KIP/planing_mds/spec.md)
- [/home/admin/KIP/planing_mds/prompt_plan.md](/home/admin/KIP/planing_mds/prompt_plan.md)
- [/home/admin/KIP/planing_mds/todo.md](/home/admin/KIP/planing_mds/todo.md)
- Quelle: [/home/admin/KIP/Abgaben/Expose.pdf](/home/admin/KIP/Abgaben/Expose.pdf)

