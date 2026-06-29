#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("benchmarks/results")
DEFAULT_UPPER_LABEL = "non-offload"
DEFAULT_BASELINE_LABEL = "offload-14GB"
THROUGHPUT_METRICS = (
    ("output_throughput", "tok/s"),
    ("request_throughput", "req/s"),
    ("tokens_per_second", "tok/s"),
    ("requests_per_second", "req/s"),
)


@dataclass(frozen=True)
class BenchmarkRun:
    label: str
    role: str
    source: Path
    throughput: float
    throughput_unit: str
    ttft_ms: float
    tpot_ms: float
    offload_gb: float | None


def _normalise(value: str) -> str:
    return value.lower().replace("_", "-")


def _label_blob(path: Path, payload: dict[str, Any]) -> str:
    candidates = [
        payload.get("dashboard_label"),
        payload.get("variant"),
        payload.get("test_name"),
        path.stem,
    ]
    return " ".join(_normalise(str(item)) for item in candidates if item is not None)


def _matches_label(path: Path, payload: dict[str, Any], label: str) -> bool:
    return _normalise(label) in _label_blob(path, payload)


def _to_float(value: Any, metric_name: str, source: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid {metric_name}: {value!r}") from exc


def _first_throughput(payload: dict[str, Any], source: Path) -> tuple[float, str]:
    for key, unit in THROUGHPUT_METRICS:
        if key in payload:
            return _to_float(payload[key], "throughput", source), unit
    joined = ", ".join(key for key, _unit in THROUGHPUT_METRICS)
    raise ValueError(f"{source} is missing throughput; expected one of: {joined}")


def _required_float(payload: dict[str, Any], key: str, metric_name: str, source: Path) -> float:
    if key in payload:
        return _to_float(payload[key], metric_name, source)
    nested_key = {
        "median_ttft_ms": ("ttft_ms", "median"),
        "median_tpot_ms": ("tpot_ms", "median"),
    }.get(key)
    if nested_key is not None:
        current: Any = payload
        for part in nested_key:
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            return _to_float(current, metric_name, source)
    raise ValueError(f"{source} is missing {metric_name}; expected key: {key}")


def _parse_offload_gb(label: str, payload: dict[str, Any]) -> float | None:
    for key in ("ascend_moe_offload_gb", "offload_gb"):
        if key in payload:
            return _to_float(payload[key], key, Path("<payload>"))

    match = re.search(r"offload[-_ ]?(\d+(?:\.\d+)?)\s*gb", label, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    return payload


def _run_from_payload(path: Path, payload: dict[str, Any], label: str, role: str) -> BenchmarkRun:
    throughput, throughput_unit = _first_throughput(payload, path)
    return BenchmarkRun(
        label=label,
        role=role,
        source=path,
        throughput=throughput,
        throughput_unit=throughput_unit,
        ttft_ms=_required_float(payload, "median_ttft_ms", "TTFT", path),
        tpot_ms=_required_float(payload, "median_tpot_ms", "TPOT", path),
        offload_gb=_parse_offload_gb(label, payload),
    )


def _find_run(results_dir: Path, label: str, role: str) -> BenchmarkRun:
    for path in sorted(results_dir.glob("*.json")):
        payload = _read_json_file(path)
        if payload is None:
            continue
        if _matches_label(path, payload, label):
            return _run_from_payload(path, payload, label, role)
    raise ValueError(f"Missing benchmark result for {label} in {results_dir}")


def load_dashboard_data(
    results_dir: str | Path,
    upper_label: str = DEFAULT_UPPER_LABEL,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
) -> tuple[BenchmarkRun, BenchmarkRun]:
    result_path = Path(results_dir)
    if not result_path.exists():
        raise ValueError(f"Results directory does not exist: {result_path}")

    upper = _find_run(result_path, upper_label, "upper_bound")
    baseline = _find_run(result_path, baseline_label, "baseline")
    return upper, baseline


def _percent(value: float) -> str:
    return f"{value:.1f}%"


def _signed_percent(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _ratio_percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("Cannot compute percentage with zero denominator")
    return numerator / denominator * 100.0


def _delta_percent(value: float, reference: float) -> float:
    if reference == 0:
        raise ValueError("Cannot compute delta with zero reference")
    return (value - reference) / reference * 100.0


def _bar_width(percent: float) -> str:
    return f"{max(0.0, min(percent, 100.0)):.1f}%"


def _fmt_number(value: float, unit: str) -> str:
    return f"{value:,.1f} {unit}"


def _offload_arg(run: BenchmarkRun) -> str:
    if run.offload_gb is None:
        return "--ascend-moe-offload-gb unknown"
    return f"--ascend-moe-offload-gb {run.offload_gb:g}"


def render_dashboard_html(upper: BenchmarkRun, baseline: BenchmarkRun) -> str:
    if upper.throughput_unit != baseline.throughput_unit:
        raise ValueError(
            "Cannot compare throughput with different units: "
            f"{upper.label} uses {upper.throughput_unit}, {baseline.label} uses {baseline.throughput_unit}"
        )

    throughput_retained = _ratio_percent(baseline.throughput, upper.throughput)
    ttft_overhead = _delta_percent(baseline.ttft_ms, upper.ttft_ms)
    tpot_overhead = _delta_percent(baseline.tpot_ms, upper.tpot_ms)

    upper_label = html.escape(upper.label)
    baseline_label = html.escape(baseline.label)
    offload_arg = html.escape(_offload_arg(baseline))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MoE Offload Performance Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #111827;
      --muted: #64748b;
      --line: #d9e2ec;
      --upper: #0f766e;
      --baseline: #f97316;
      --accent: #2563eb;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .tag {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--muted);
      white-space: nowrap;
      font-size: 13px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .metric-note {{
      color: var(--muted);
      font-size: 12px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: middle;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #eef3f8;
      color: #475569;
      font-size: 12px;
      text-transform: uppercase;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .role {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }}
    .bars {{
      padding: 16px 14px 18px;
      display: grid;
      gap: 14px;
    }}
    .bar-label {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .track {{
      height: 10px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .fill.upper {{
      background: var(--upper);
    }}
    .fill.baseline {{
      background: var(--baseline);
    }}
    .fill.accent {{
      background: var(--accent);
    }}
    @media (max-width: 760px) {{
      header {{
        display: block;
      }}
      .tag {{
        display: inline-block;
        margin-top: 12px;
        white-space: normal;
      }}
      .summary {{
        grid-template-columns: 1fr;
      }}
      th, td {{
        padding: 10px 8px;
        font-size: 13px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>MoE Offload Performance Dashboard</h1>
        <p>MoE non-offloading upper bound is compared against the current offload baseline using {offload_arg}.</p>
      </div>
      <div class="tag">Data source: benchmarks/results/*.json</div>
    </header>

    <section class="summary" aria-label="Metric summary">
      <div class="metric">
        <div class="metric-label">Throughput retained</div>
        <div class="metric-value">{_percent(throughput_retained)}</div>
        <div class="metric-note">higher is better; baseline divided by upper bound</div>
      </div>
      <div class="metric">
        <div class="metric-label">TTFT overhead</div>
        <div class="metric-value">{_signed_percent(ttft_overhead)}</div>
        <div class="metric-note">lower is better; baseline relative to upper bound</div>
      </div>
      <div class="metric">
        <div class="metric-label">TPOT overhead</div>
        <div class="metric-value">{_signed_percent(tpot_overhead)}</div>
        <div class="metric-note">lower is better; baseline relative to upper bound</div>
      </div>
    </section>

    <section class="panel" aria-label="Benchmark results table">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Throughput</th>
            <th>TTFT</th>
            <th>TPOT</th>
            <th>Comparison</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{upper_label}<span class="role">MoE non-offloading upper bound</span></td>
            <td>{_fmt_number(upper.throughput, upper.throughput_unit)}</td>
            <td>{_fmt_number(upper.ttft_ms, "ms")}</td>
            <td>{_fmt_number(upper.tpot_ms, "ms")}</td>
            <td>100.0%</td>
          </tr>
          <tr>
            <td>{baseline_label}<span class="role">Offload baseline, {offload_arg}</span></td>
            <td>{_fmt_number(baseline.throughput, baseline.throughput_unit)}</td>
            <td>{_fmt_number(baseline.ttft_ms, "ms")}</td>
            <td>{_fmt_number(baseline.tpot_ms, "ms")}</td>
            <td>{_percent(throughput_retained)} throughput, {_signed_percent(ttft_overhead)} TTFT, {_signed_percent(tpot_overhead)} TPOT</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel bars" aria-label="Comparison bars">
      <div>
        <div class="bar-label"><span>Throughput: offload baseline retained</span><strong>{_percent(throughput_retained)}</strong></div>
        <div class="track"><div class="fill baseline" style="width: {_bar_width(throughput_retained)}"></div></div>
      </div>
      <div>
        <div class="bar-label"><span>TTFT overhead versus upper bound</span><strong>{_signed_percent(ttft_overhead)}</strong></div>
        <div class="track"><div class="fill accent" style="width: {_bar_width(abs(ttft_overhead))}"></div></div>
      </div>
      <div>
        <div class="bar-label"><span>TPOT overhead versus upper bound</span><strong>{_signed_percent(tpot_overhead)}</strong></div>
        <div class="track"><div class="fill accent" style="width: {_bar_width(abs(tpot_overhead))}"></div></div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def write_dashboard(
    results_dir: str | Path,
    output: str | Path,
    upper_label: str = DEFAULT_UPPER_LABEL,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
) -> Path:
    upper, baseline = load_dashboard_data(results_dir, upper_label, baseline_label)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard_html(upper, baseline), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a MoE offload performance dashboard from benchmark JSON.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing benchmark JSON files. Defaults to benchmarks/results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <results-dir>/moe_offload_dashboard.html.",
    )
    parser.add_argument(
        "--upper-label",
        default=DEFAULT_UPPER_LABEL,
        help="Label used to identify the non-offloading upper-bound result.",
    )
    parser.add_argument(
        "--baseline-label",
        default=DEFAULT_BASELINE_LABEL,
        help="Label used to identify the offload baseline result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.results_dir / "moe_offload_dashboard.html"
    output_path = write_dashboard(
        results_dir=args.results_dir,
        output=output,
        upper_label=args.upper_label,
        baseline_label=args.baseline_label,
    )
    print(output_path)


if __name__ == "__main__":
    main()
