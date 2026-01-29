#!/usr/bin/env python3
"""
Batch driver to run CPD Online (one-sided CUSUM) over a token_stats CSV.

For every row with a token_stream payload we extract the post-prefix entropies,
run the online CUSUM detector using the prefix (system-prompt) entropies as a
robust baseline, and dump per-row metrics to a CSV for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import cpd_online
from .cpd_types import OnlineCUSUMConfig


def _extract_entropies(token_stream: str) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[int]]:
    """
    Parse the token_stream JSON and return:
        - prefix entropies (np.ndarray) for tokens preceding the user prompt
        - entropies (np.ndarray) for post-prefix tokens
        - suffix_start (post-prefix index) if available
        - suffix_len (token count in suffix) if available
    """
    payload = json.loads(token_stream)
    tokens = payload.get("tokens") or []
    meta = payload.get("meta") or {}

    ent_list: List[float] = []
    prefix_list: List[float] = []
    for tok in tokens:
        ent = tok.get("entropy")
        if ent is None or (isinstance(ent, float) and math.isnan(ent)):
            continue
        if tok.get("pos_postprefix") is None:
            prefix_list.append(float(ent))
        else:
            ent_list.append(float(ent))

    suffix_start = meta.get("suffix_start_postprefix")
    if suffix_start is not None:
        try:
            suffix_start = int(suffix_start)
        except (TypeError, ValueError):
            suffix_start = None

    suffix_len = meta.get("suffix_len_tokens_ctx")
    if suffix_len is not None:
        try:
            suffix_len = int(suffix_len)
        except (TypeError, ValueError):
            suffix_len = None

    return (
        np.asarray(prefix_list, dtype=float),
        np.asarray(ent_list, dtype=float),
        suffix_start,
        suffix_len,
    )


def _online_delay(t_alarm: Optional[int], suffix_start: Optional[int]) -> Optional[int]:
    if t_alarm is None or suffix_start is None:
        return None
    return int(t_alarm - suffix_start)


def _run_online(ent: np.ndarray, cfg: OnlineCUSUMConfig, baseline: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if ent.size == 0:
        return {
            "online_alarm": 0,
            "online_t_alarm": None,
            "online_final_W_plus": 0.0,
            "online_final_W_minus": 0.0,
            "online_max_W_plus": 0.0,
            "online_max_Z": 0.0,
            "online_W_plus_trace": json.dumps([]),
        }
    baseline_seq = baseline if baseline is not None and baseline.size > 0 else None
    if baseline_seq is None:
        raise ValueError(
            "Missing prefix entropy baseline. This driver uses the system-prompt (prefix) tokens for baseline; "
            "regenerate token stats with prefix tokens included."
        )
    cfg_local = replace(cfg, baseline_window=max(1, int(baseline_seq.size)))

    state, events = cpd_online.run_full(ent, cfg_local, baseline=baseline_seq)
    max_w_plus = max((ev.W_plus for ev in events), default=0.0)
    max_z = max((abs(ev.z_t) for ev in events), default=0.0)
    w_plus_trace = [float(ev.W_plus) for ev in events]
    return {
        "online_alarm": int(state.alarm),
        "online_t_alarm": state.T_alarm,
        "online_final_W_plus": state.W_plus,
        "online_final_W_minus": state.W_minus,
        "online_max_W_plus": max_w_plus,
        "online_max_Z": max_z,
        "online_W_plus_trace": json.dumps(w_plus_trace),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch CPD Online over token_stats CSV.")
    p.add_argument("--stats-csv", required=True, help="stats/<model>_token_stats(_tok_id).csv with token_stream column")
    p.add_argument("--out-csv", required=True, help="Destination CSV to write per-row CPD metrics")
    p.add_argument("--online-k", type=float, default=0.5, help="CUSUM slack parameter k")
    p.add_argument("--online-h", type=float, default=10.0, help="CUSUM threshold h")
    p.add_argument("--online-reset-after-alarm", action="store_true", help="Reset statistics after alarm")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows to process (debug)")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    try:
        df = pd.read_csv(args.stats_csv)
    except Exception as exc:
        print(f"Failed to read {args.stats_csv}: {exc}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"token_stream"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Missing required columns in stats CSV: {sorted(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("No rows left after filtering; aborting.", file=sys.stderr)
        sys.exit(1)

    online_cfg = OnlineCUSUMConfig(
        k=args.online_k,
        h=args.online_h,
        baseline_window=1,
        reset_after_alarm=args.online_reset_after_alarm,
    )

    rows: List[Dict[str, object]] = []
    total = len(df) if args.max_rows is None else min(len(df), args.max_rows)

    use_tqdm = False
    if total > 50:
        try:
            from tqdm import tqdm  # type: ignore
            progress_iter = tqdm(range(total), desc="CPD batch", unit="row")
            use_tqdm = True
        except Exception:
            progress_iter = range(total)
    else:
        progress_iter = range(total)

    for idx in progress_iter:
        row = df.iloc[idx]
        token_stream = row.get("token_stream")
        if not isinstance(token_stream, str):
            continue

        if not use_tqdm and (idx % 10 == 0 or idx == total - 1):
            print(f"[INFO] Processing row {idx + 1}/{total}", flush=True)

        prefix_ent, ent, suffix_start, suffix_len = _extract_entropies(token_stream)

        try:
            online_res = _run_online(ent, online_cfg, baseline=prefix_ent)
        except Exception as exc:
            print(
                f"[ERROR] CPD Online failed for row_index={int(df.index[idx])}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        online_delay = _online_delay(online_res["online_t_alarm"], suffix_start)

        rec: Dict[str, object] = {
            "row_index": int(df.index[idx]),
            "algorithm": row.get("algorithm"),
            "is_adversarial": row.get("is_adversarial"),
            "suffix_start_postprefix": suffix_start,
            "suffix_len_tokens": suffix_len,
            "num_tokens_postprefix": int(ent.size),
            "num_tokens_prefix": int(prefix_ent.size),
        }
        rec.update(online_res)
        rec["online_alarm_delay"] = online_delay

        rows.append(rec)

    if use_tqdm:
        progress_iter.close()

    if not rows:
        print("No rows processed; did the CSV contain token_stream entries?", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
