# SPDX-License-Identifier: Apache-2.0
"""Compare smoke output artifacts for fixed-slot correctness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_outputs_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = str(record.get("request_id", ""))
            if not request_id:
                raise ValueError(f"missing request_id in {path}:{line_number}")
            if request_id in seen_request_ids:
                raise ValueError(f"duplicate request_id in {path}: {request_id}")
            seen_request_ids.add(request_id)
            records.append(record)
    return records


def compare_output_records(
    baseline_outputs: list[dict[str, Any]],
    candidate_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(record["request_id"]): record for record in baseline_outputs}
    candidate_by_id = {str(record["request_id"]): record for record in candidate_outputs}

    mismatches: list[dict[str, Any]] = []
    matched = 0
    for request_id, baseline in baseline_by_id.items():
        candidate = candidate_by_id.get(request_id)
        if candidate is None:
            mismatches.append({"request_id": request_id, "reason": "missing candidate output"})
            continue
        baseline_token_ids = [int(token_id) for token_id in baseline.get("output_token_ids", [])]
        candidate_token_ids = [int(token_id) for token_id in candidate.get("output_token_ids", [])]
        if baseline_token_ids == candidate_token_ids:
            matched += 1
            continue
        mismatches.append(
            {
                "request_id": request_id,
                "reason": "token_ids differ",
                "baseline_output_tokens": len(baseline_token_ids),
                "candidate_output_tokens": len(candidate_token_ids),
            }
        )

    extra_request_ids = sorted(set(candidate_by_id) - set(baseline_by_id))
    for request_id in extra_request_ids:
        mismatches.append({"request_id": request_id, "reason": "extra candidate output"})

    missing = sum(1 for mismatch in mismatches if mismatch["reason"] == "missing candidate output")
    extra = len(extra_request_ids)
    mismatched = sum(1 for mismatch in mismatches if mismatch["reason"] == "token_ids differ")
    return {
        "status": "ok" if not mismatches else "failed",
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
    }


def write_comparison_summary(
    *,
    baseline_outputs: list[dict[str, Any]],
    candidate_outputs: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    summary = compare_output_records(baseline_outputs, candidate_outputs)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_comparison_summary(
        baseline_outputs=load_outputs_jsonl(args.baseline),
        candidate_outputs=load_outputs_jsonl(args.candidate),
        output_path=args.output,
    )
    print("SMOKE_OUTPUT_COMPARISON " + json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
