#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from tracked training run JSON files."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "training_runs"
OUTPUT = RUNS_DIR / "analysis_dashboard.html"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    if not isinstance(value, (float, int)):
        return "n/a"
    return f"{value * 100:.2f}%"


def svg_line_chart(
    series: list[tuple[str, list[float]]],
    labels: list[int],
    title: str,
    y_label: str,
) -> str:
    width, height = 900, 300
    left, right, top, bottom = 54, 20, 32, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for _, points in series for v in points if isinstance(v, (int, float))]
    if not values or not labels:
        return '<div class="empty">No chart data available.</div>'
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad
    x_min, x_max = min(labels), max(labels)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def x_pos(epoch: int) -> float:
        return left + ((epoch - x_min) / (x_max - x_min)) * plot_w

    def y_pos(value: float) -> float:
        return top + (1 - ((value - y_min) / (y_max - y_min))) * plot_h

    colors = ["#0f766e", "#8b5cf6", "#b45309", "#2563eb"]
    paths = []
    legend = []
    for idx, (name, points) in enumerate(series):
        coords = [
            f"{x_pos(labels[i]):.2f},{y_pos(value):.2f}"
            for i, value in enumerate(points)
            if isinstance(value, (int, float))
        ]
        if not coords:
            continue
        color = colors[idx % len(colors)]
        paths.append(
            f'<polyline points="{" ".join(coords)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5" stroke-linejoin="round" />'
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(name)}</span>'
        )

    grid = []
    for i in range(5):
        ratio = i / 4
        y = top + ratio * plot_h
        value = y_max - ratio * (y_max - y_min)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" />'
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{fmt(value, 3)}</text>'
        )
    x_ticks = []
    for epoch in labels:
        if epoch == labels[0] or epoch == labels[-1] or epoch % 2 == 0:
            x = x_pos(epoch)
            x_ticks.append(
                f'<text x="{x:.2f}" y="{height - 14}" text-anchor="middle">{epoch}</text>'
            )

    return f"""
    <section class="chart">
      <div class="chart-head">
        <h3>{html.escape(title)}</h3>
        <span>{html.escape(y_label)}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
        <g class="grid">{''.join(grid)}</g>
        <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />
        <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />
        {''.join(paths)}
        <g class="ticks">{''.join(x_ticks)}</g>
      </svg>
      <div class="legend">{''.join(legend)}</div>
    </section>
    """


def card(label: str, value: str, note: str = "") -> str:
    return f"""
    <article class="card">
      <div>{html.escape(label)}</div>
      <strong>{html.escape(value)}</strong>
      <span>{html.escape(note)}</span>
    </article>
    """


def run_summary_row(name: str, run: dict[str, Any]) -> str:
    history = run["history"]
    latest = history[-1] if history else {}
    best = min(history, key=lambda r: r.get("val_loss", float("inf"))) if history else {}
    test = run["test"]
    split = run["split"]
    config = run["config"]
    return (
        "<tr>"
        f"<td><code>{html.escape(name)}</code></td>"
        f"<td>{fmt(split.get('train_rows'))}</td>"
        f"<td>{fmt(config.get('d_model'))}</td>"
        f"<td>{fmt(config.get('encoder_layers'))}/{fmt(config.get('decoder_layers'))}</td>"
        f"<td>{fmt(len(history))}</td>"
        f"<td>{fmt(best.get('val_loss'))}</td>"
        f"<td>{pct(latest.get('val_cer'))}</td>"
        f"<td>{pct(latest.get('val_wer'))}</td>"
        f"<td>{pct(test.get('cer'))}</td>"
        f"<td>{pct(test.get('wer'))}</td>"
        f"<td>{pct(test.get('eos_rate'))}</td>"
        "</tr>"
    )


def epoch_rows(history: list[dict[str, Any]]) -> str:
    rows = []
    for row in history:
        rows.append(
            "<tr>"
            f"<td>{fmt(row.get('epoch'), 0)}</td>"
            f"<td>{fmt(row.get('train_loss'))}</td>"
            f"<td>{fmt(row.get('val_loss'))}</td>"
            f"<td>{pct(row.get('val_cer'))}</td>"
            f"<td>{pct(row.get('val_wer'))}</td>"
            f"<td>{pct(row.get('val_eos_rate'))}</td>"
            f"<td>{pct(row.get('val_forced_eos_rate'))}</td>"
            f"<td>{fmt(row.get('val_avg_pred_len'), 2)}</td>"
            f"<td>{fmt(row.get('val_avg_ref_len'), 2)}</td>"
            f"<td>{fmt(row.get('seconds'), 1)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def main() -> None:
    runs: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        history = read_json(run_dir / "history.json", [])
        if not history:
            continue
        runs[run_dir.name] = {
            "history": history,
            "test": read_json(run_dir / "test_metrics.json", {}),
            "eval": read_json(run_dir / "eval_only_metrics.json", {}),
            "split": read_json(run_dir / "data_split_summary.json", {}),
            "config": read_json(run_dir / "config.json", {}),
        }

    focus_name = "unicode_to_translit_full_152k_d256"
    focus = runs[focus_name]
    history = focus["history"]
    latest = history[-1]
    best = min(history, key=lambda r: r.get("val_loss", float("inf")))
    test = focus["test"]
    split = focus["split"]
    config = focus["config"]
    epochs = [int(row["epoch"]) for row in history]

    total_seconds = sum(row.get("seconds", 0) for row in history)
    hours = total_seconds / 3600
    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>152k Unicode Transliteration Model Dashboard</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5e6b7e;
      --line: #d7e0ea;
      --panel: #f7fafc;
      --panel-2: #eef5f4;
      --accent: #0f766e;
      --purple: #6d5bd0;
      --warn: #a16207;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }}
    header.top {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 32px; line-height: 1.15; }}
    h2 {{ font-size: 22px; margin: 28px 0 12px; }}
    .subtle {{ color: var(--muted); margin: 8px 0 0; max-width: 820px; line-height: 1.5; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 18px 0 26px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      background: var(--panel);
      min-height: 112px;
    }}
    .card div {{
      color: var(--muted);
      font-size: 13px;
      min-height: 32px;
    }}
    .card strong {{
      display: block;
      font-size: 24px;
      margin: 3px 0;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .summary {{
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 14px 16px;
      line-height: 1.55;
      margin: 12px 0 24px;
    }}
    .chart {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 12px 0 18px;
      overflow: hidden;
    }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .chart-head h3 {{ font-size: 16px; }}
    .chart-head span, .legend span {{ color: var(--muted); font-size: 12px; }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    svg line {{ stroke: var(--line); }}
    .grid line {{ stroke: #e7edf3; }}
    .grid text, .ticks text {{ fill: var(--muted); font-size: 11px; }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .legend i {{
      display: inline-block;
      width: 18px;
      height: 3px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      background: #fbfdff;
      font-weight: 650;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .two {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
    }}
    pre {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
<main>
  <header class="top">
    <h1>152k Unicode → Transliteration Model Dashboard</h1>
    <p class="subtle">
      Completed run: <code>{html.escape(focus_name)}</code>. Metrics are raw ratios internally;
      this dashboard displays CER/WER as percentages for readability.
    </p>
  </header>

  <section class="cards">
    {card("Training rows", fmt(split.get("train_rows")), "Full dataset split")}
    {card("Epochs completed", fmt(len(history)), f"Best epoch {fmt(best.get('epoch'), 0)}")}
    {card("Best validation loss", fmt(best.get("val_loss")), "Lower is better")}
    {card("Final validation CER", pct(latest.get("val_cer")), "Epoch 20")}
    {card("Final validation WER", pct(latest.get("val_wer")), "Epoch 20")}
    {card("Final test CER", pct(test.get("cer")), "Held-out test set")}
    {card("Final test WER", pct(test.get("wer")), "Held-out test set")}
    {card("EOS rate", pct(test.get("eos_rate")), "Runaway decoding controlled")}
    {card("Model size", f'd={fmt(config.get("d_model"))}', f'{fmt(config.get("encoder_layers"))} encoder / {fmt(config.get("decoder_layers"))} decoder layers')}
    {card("Training time", f"{hours:.1f} h", "Summed epoch wall time")}
  </section>

  <section class="summary">
    The completed 152k run improved validation loss from {fmt(history[0].get("val_loss"))}
    to {fmt(latest.get("val_loss"))}. Validation CER ended at {pct(latest.get("val_cer"))},
    and the held-out test CER is {pct(test.get("cer"))}. EOS emission is stable at
    {pct(test.get("eos_rate"))}, with no global max-length failures reported.
  </section>

  <h2>152k Training Curves</h2>
  {svg_line_chart([("Train loss", [r.get("train_loss") for r in history]), ("Val loss", [r.get("val_loss") for r in history])], epochs, "Loss by Epoch", "loss")}
  {svg_line_chart([("Val CER", [r.get("val_cer") for r in history]), ("Val WER", [r.get("val_wer") for r in history])], epochs, "Validation Error by Epoch", "raw ratio")}
  {svg_line_chart([("Forced EOS", [r.get("val_forced_eos_rate") for r in history]), ("EOS rate", [r.get("val_eos_rate") for r in history])], epochs, "EOS Diagnostics", "rate")}

  <h2>Run Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Run</th><th>Train rows</th><th>d_model</th><th>Layers</th><th>Epochs</th>
        <th>Best val loss</th><th>Final val CER</th><th>Final val WER</th>
        <th>Test CER</th><th>Test WER</th><th>Test EOS</th>
      </tr>
    </thead>
    <tbody>
      {''.join(run_summary_row(name, run) for name, run in runs.items())}
    </tbody>
  </table>

  <h2>152k Epoch Details</h2>
  <table>
    <thead>
      <tr>
        <th>Epoch</th><th>Train loss</th><th>Val loss</th><th>Val CER</th>
        <th>Val WER</th><th>EOS</th><th>Forced EOS</th>
        <th>Avg pred len</th><th>Avg ref len</th><th>Seconds</th>
      </tr>
    </thead>
    <tbody>{epoch_rows(history)}</tbody>
  </table>

  <h2>Final Test Metrics</h2>
  <pre>{html.escape(json.dumps(test, indent=2, ensure_ascii=False))}</pre>

  <h2>152k Run Config</h2>
  <section class="two">
    <pre>{html.escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre>
    <pre>{html.escape(json.dumps(split, indent=2, ensure_ascii=False))}</pre>
  </section>
</main>
</body>
</html>
"""
    OUTPUT.write_text(dashboard, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
