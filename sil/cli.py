"""Command-line interface.

    python -m sil analyze path/to/transcript.json [--out chart.png] [--config path]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .loader import TranscriptError, load_transcript
from .metrics import Config


def _load_config(path: str | None) -> Config:
    if path is None:
        return Config()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(raw) - valid
    if unknown:
        raise SystemExit(
            f"Unknown config keys: {sorted(unknown)}; valid keys: {sorted(valid)}"
        )
    return Config(**raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sil",
        description="Analyze a human-AI conversation transcript for failure modes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a transcript JSON file")
    analyze.add_argument("transcript", help="Path to the transcript JSON file")
    analyze.add_argument(
        "--out", default="chart.png", help="Output chart path (default: chart.png)"
    )
    analyze.add_argument(
        "--config", default=None, help="Path to a JSON file overriding Config defaults"
    )

    args = parser.parse_args(argv)

    try:
        transcript = load_transcript(args.transcript)
    except TranscriptError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    cfg = _load_config(args.config)

    # Imported late so `sil analyze` fails fast on transcript errors without
    # paying the model-load cost first.
    from . import metrics
    from .embed import embed_text, embed_turns
    from .viz import render, summarize

    E = embed_turns(transcript.turns, session_id=transcript.session_id)
    goal_embedding = embed_text(transcript.goal)

    try:
        novelty_s = metrics.novelty(E, window=cfg.novelty_window)
        drift_s = metrics.drift(E, goal_embedding, alpha=cfg.drift_alpha)
        agreement_s = metrics.agreement_density(
            transcript.turns, window=cfg.agreement_window
        )
        flags = metrics.flag_failures(novelty_s, drift_s, agreement_s, cfg)
    except NotImplementedError as e:
        print(
            f"error: {e}\n"
            "The detection engine (sil/metrics.py) is not implemented yet.",
            file=sys.stderr,
        )
        return 2

    render(transcript, novelty_s, drift_s, agreement_s, flags, args.out)
    print(summarize(flags))
    print(f"chart saved to {args.out}")
    return 0
