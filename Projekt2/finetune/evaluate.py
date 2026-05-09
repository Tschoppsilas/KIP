"""Vergleich: Fine-Tuned Modell vs. Baseline (yolo11n.pt).

Wertet Precision, Recall und mAP50 auf dem Validierungsset aus und
gibt eine Zusammenfassung aus.

Aufruf:
    python finetune/evaluate.py --model finetune/runs/train/weights/best.pt
    python finetune/evaluate.py --model finetune/runs/train/weights/best.pt --baseline yolo11s.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML    = PROJECT_ROOT / "finetune" / "data.yaml"


def evaluate_model(model_path: str, data_yaml: str, label: str) -> dict:
    """Führt YOLO-Validierung durch und gibt Metriken zurück."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=False)
    results = {
        "label":      label,
        "model":      model_path,
        "precision":  float(metrics.box.mp),
        "recall":     float(metrics.box.mr),
        "map50":      float(metrics.box.map50),
        "map50_95":   float(metrics.box.map),
    }
    return results


def print_comparison(baseline: dict, finetuned: dict) -> None:
    """Gibt eine formatierte Vergleichstabelle aus."""
    print("\n" + "=" * 60)
    print(f"{'Metrik':<20} {'Baseline':>15} {'Fine-Tuned':>15} {'Δ':>8}")
    print("-" * 60)
    for key in ("precision", "recall", "map50", "map50_95"):
        b = baseline[key]
        f = finetuned[key]
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        print(f"{key:<20} {b:>15.4f} {f:>15.4f} {sign}{delta:>7.4f}")
    print("=" * 60)

    improved = finetuned["map50"] > baseline["map50"]
    verdict = "✓ Fine-Tuned ist BESSER (mAP50)" if improved else "✗ Kein Fortschritt – Baseline überlegen"
    print(f"\nErgebnis: {verdict}")

    # Modellpfad für visualize_pipeline.py ausgeben
    best = Path(finetuned["model"])
    if best.exists():
        print(f"\nModell-Pfad für Pipeline:")
        print(f"  python visualize_pipeline.py Videos/Muenchenstein_1.mp4 90 output_finetuned.mp4")
        print(f"  → Ändere MODEL = \"{best}\" in visualize_pipeline.py")
        print(f"  → Ändere MODEL = \"{best}\" in tactic_board_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True,         help="Pfad zu best.pt")
    parser.add_argument("--baseline", default="yolo11n.pt",  help="Baseline-Modell")
    parser.add_argument("--data",     default=str(DATA_YAML))
    args = parser.parse_args()

    print(f"Evaluiere Baseline ({args.baseline}) …")
    base_metrics = evaluate_model(args.baseline, args.data, "Baseline")

    print(f"Evaluiere Fine-Tuned ({args.model}) …")
    ft_metrics   = evaluate_model(args.model,    args.data, "Fine-Tuned")

    print_comparison(base_metrics, ft_metrics)

    # Ergebnis in JSON speichern
    import json
    out = PROJECT_ROOT / "finetune" / "eval_results.json"
    out.write_text(json.dumps({"baseline": base_metrics, "finetuned": ft_metrics}, indent=2))
    print(f"\nErgebnisse gespeichert: {out}")
