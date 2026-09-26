"""Command-line interface.

    python -m sil analyze CHAT [--goal TEXT] [--conversation N|TITLE]
                               [--out chart.png] [--config path]
    python -m sil list CHAT

CHAT can be stuck-in-the-loop JSON, a ChatGPT or Claude data export
(conversations.json), or pasted text/markdown with speaker labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .formats import list_conversations, load_any
from .loader import TranscriptError
from .metrics import Config


def _load_config(path: str | None) -> Config:
    if path is None:
        return Config()
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(raw) - valid
    if unknown:
        raise SystemExit(
            f"Unknown config keys: {sorted(unknown)}; valid keys: {sorted(valid)}"
        )
    return Config(**raw)


def _cmd_list(args) -> int:
    try:
        convs = list_conversations(args.chat)
    except TranscriptError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for c in convs:
        print(f"{c.index:>4}  {c.n_turns:>4} turns  {c.title}")
    return 0


def _goal_embedding(transcript, E, metrics, embed_text):
    if transcript.goal is not None:
        return embed_text(transcript.goal)
    infer_goal = getattr(metrics, "infer_goal", None)
    if infer_goal is not None:
        try:
            return infer_goal(transcript.turns, E)
        except NotImplementedError:
            pass
    return None


def _cmd_analyze(args) -> int:
    try:
        transcript = load_any(args.chat, conversation=args.conversation, goal=args.goal)
    except TranscriptError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    cfg = _load_config(args.config)

    # Imported late so bad input fails fast without paying the model-load cost.
    from . import metrics
    from .embed import embed_text, embed_turns
    from .viz import render, summarize

    E = embed_turns(transcript.turns, session_id=transcript.session_id)
    goal_embedding = _goal_embedding(transcript, E, metrics, embed_text)
    if goal_embedding is None:
        print(
            "error: this chat has no goal. Pass one with --goal \"...\" "
            "(goal inference isn't implemented yet).",
            file=sys.stderr,
        )
        return 1

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
            "The detection engine (sil/metrics.py) is not finished yet.",
            file=sys.stderr,
        )
        return 2

    render(transcript, novelty_s, drift_s, agreement_s, flags, args.out)
    print(summarize(flags))
    print(f"chart saved to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sil",
        description="Analyze a human-AI conversation for failure modes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one conversation")
    analyze.add_argument(
        "chat",
        help="sil JSON, ChatGPT/Claude export (conversations.json), or pasted .txt/.md",
    )
    analyze.add_argument("--goal", default=None, help="What the session was trying to achieve")
    analyze.add_argument(
        "--conversation",
        default=None,
        help="Which conversation in an export file: 0-based index or part of its title",
    )
    analyze.add_argument(
        "--out", default="chart.png", help="Output chart path (default: chart.png)"
    )
    analyze.add_argument(
        "--config", default=None, help="Path to a JSON file overriding Config defaults"
    )

    lst = sub.add_parser("list", help="List the conversations in a file")
    lst.add_argument("chat")

    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    return _cmd_analyze(args)
