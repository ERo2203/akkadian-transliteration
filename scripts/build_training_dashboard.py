#!/usr/bin/env python3
"""Build a self-contained HTML dashboard for a training run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_card(label: str, value: Any, hint: str = "") -> str:
    return f"""
    <section class="metric">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(fmt(value))}</div>
      <div class="metric-hint">{html.escape(hint)}</div>
    </section>
    """


def history_rows(history: List[Dict[str, Any]]) -> str:
    rows = []
    for row in history:
        rows.append(
            "<tr>"
            f"<td>{row.get('epoch')}</td>"
            f"<td>{fmt(row.get('train_loss'))}</td>"
            f"<td>{fmt(row.get('val_loss'))}</td>"
            f"<td>{fmt(row.get('val_cer'))}</td>"
            f"<td>{fmt(row.get('val_wer'))}</td>"
            f"<td>{fmt(row.get('val_eos_rate'))}</td>"
            f"<td>{fmt(row.get('val_forced_eos_rate'))}</td>"
            f"<td>{fmt(row.get('val_avg_max_char_run'), 2)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def prediction_blocks(predictions: List[Dict[str, str]]) -> str:
    blocks = []
    for idx, row in enumerate(predictions[:20], 1):
        blocks.append(
            f"""
            <article class="prediction">
              <header>
                <strong>Sample {idx}</strong>
                <span>CER {html.escape(row.get('cer', ''))}</span>
                <span>WER {html.escape(row.get('wer', ''))}</span>
                <span>len {html.escape(row.get('pred_len', ''))}/{html.escape(row.get('ref_len', ''))}</span>
                <span>limit {html.escape(row.get('decode_limit', ''))}</span>
                <span>EOS {html.escape(row.get('eos_hit', ''))}</span>
                <span>run {html.escape(row.get('max_char_run', ''))}</span>
              </header>
              <h3>Reference</h3>
              <pre>{html.escape(row.get('ref', ''))}</pre>
              <h3>Prediction</h3>
              <pre>{html.escape(row.get('pred', ''))}</pre>
            </article>
            """
        )
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="checkpoints/unicode_to_translit_eos_weighted")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "dashboard.html"
    history = read_json(run_dir / "history.json", [])
    test_metrics = read_json(run_dir / "test_metrics.json", {})
    eval_metrics = read_json(run_dir / "eval_only_metrics.json", {})
    split = read_json(run_dir / "data_split_summary.json", {})
    config = read_json(run_dir / "config.json", {})
    predictions = read_json(run_dir / "prediction_samples.json", [])
    latest = history[-1] if history else {}

    metrics = eval_metrics or test_metrics
    cards = "\n".join(
        [
            metric_card("Validation CER", latest.get("val_cer", "n/a"), "lower is better; raw ratio"),
            metric_card("Validation WER", latest.get("val_wer", "n/a"), "lower is better; raw ratio"),
            metric_card("Forced EOS Rate", latest.get("val_forced_eos_rate", metrics.get("forced_eos_rate", "n/a")), "share stopped by decode cap"),
            metric_card("EOS Rate", latest.get("val_eos_rate", metrics.get("eos_rate", "n/a")), "share with EOS token"),
            metric_card("Test CER", test_metrics.get("cer", "n/a"), "held-out raw ratio"),
            metric_card("Test WER", test_metrics.get("wer", "n/a"), "held-out raw ratio"),
            metric_card("Avg Max Char Run", metrics.get("avg_max_char_run", latest.get("val_avg_max_char_run", "n/a")), "repetition indicator"),
            metric_card("Rows", f"{split.get('train_rows', 'n/a')} train", "current run size"),
        ]
    )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unicode Transliteration Training Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #607086;
      --line: #d9e1ea;
      --panel: #f8fafc;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    .subtle {{
      color: var(--muted);
      margin: 0 0 24px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0 28px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .metric-hint {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 28px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 9px 8px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #fbfdff;
    }}
    .prediction {{
      border-top: 1px solid var(--line);
      padding: 16px 0;
    }}
    .prediction header {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 10px;
    }}
    .prediction header span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    h3 {{
      margin: 10px 0 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
    }}
    .config {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Unicode Transliteration Training Dashboard</h1>
    <p class="subtle">Run directory: {html.escape(str(run_dir))}</p>
    <section class="metrics">{cards}</section>

    <h2>Epoch History</h2>
    <table>
      <thead>
        <tr>
          <th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Val CER</th>
          <th>Val WER</th><th>EOS</th><th>Forced EOS</th><th>Avg Run</th>
        </tr>
      </thead>
      <tbody>{history_rows(history)}</tbody>
    </table>

    <h2>Prediction Samples</h2>
    {prediction_blocks(predictions)}

    <h2>Run Config</h2>
    <section class="config">
      <pre>{html.escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre>
      <pre>{html.escape(json.dumps(split, indent=2, ensure_ascii=False))}</pre>
    </section>
  </main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
