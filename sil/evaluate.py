"""Score the detector against hand labels.

Reads eval/labels.csv (one row per conversation, 1 / 0 / ? per failure mode)
and eval/split.json, runs the detector on one half of the sample, and reports
precision / recall / F1 per mode.

The calibration half is for tuning and can be evaluated as often as needed.
Every evaluation of the test half is appended to eval/test_log.csv, because
each look at test results makes them a little less of a test.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .loader import Transcript, load_transcript
from .metrics import Config

EVALUATED_MODES = ["validation_loop", "premature_convergence", "drift"]


@dataclass
class ModeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    skipped: int = 0  # labeled "?"

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


@dataclass
class EvalResult:
    split: str
    n_chats: int
    goals_from_first_message: int
    modes: dict[str, ModeScore] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)


def read_labels(eval_dir: str) -> list[dict]:
    with open(Path(eval_dir) / "labels.csv", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_split(eval_dir: str) -> dict[str, str]:
    return json.loads((Path(eval_dir) / "split.json").read_text(encoding="utf-8"))["split"]


def is_labeled(row: dict) -> bool:
    return any((row.get(m) or "").strip() for m in EVALUATED_MODES)


def default_detect(t: Transcript, goal: str, cfg: Config) -> set[str]:
    from . import metrics
    from .embed import embed_text, embed_turns

    E = embed_turns(t.turns, session_id=t.session_id)
    flags = metrics.flag_failures(
        metrics.novelty(E, window=cfg.novelty_window),
        metrics.drift(E, embed_text(goal), alpha=cfg.drift_alpha),
        metrics.agreement_density(t.turns, window=cfg.agreement_window),
        cfg,
    )
    return {f.mode for f in flags}


def evaluate(
    eval_dir: str,
    split: str = "calibration",
    cfg: Config | None = None,
    detect: Callable[[Transcript, str, Config], set[str]] = default_detect,
) -> EvalResult:
    cfg = cfg or Config()
    assignment = read_split(eval_dir)
    rows = [r for r in read_labels(eval_dir) if assignment.get(r["id"]) == split and is_labeled(r)]

    result = EvalResult(split=split, n_chats=len(rows), goals_from_first_message=0)
    result.modes = {m: ModeScore() for m in EVALUATED_MODES}

    for row in rows:
        t = load_transcript(str(Path(eval_dir) / "data" / f"{row['id']}.json"))
        goal = (row.get("goal") or "").strip()
        if not goal:
            goal = next(x.text for x in t.turns if x.role == "user")
            result.goals_from_first_message += 1
        predicted = detect(t, goal, cfg)

        detail = {"id": row["id"], "predicted": sorted(predicted), "labeled": []}
        for m in EVALUATED_MODES:
            label = (row.get(m) or "").strip()
            score = result.modes[m]
            if label not in ("0", "1"):
                score.skipped += 1
                continue
            truth, pred = label == "1", m in predicted
            if truth:
                detail["labeled"].append(m)
            if truth and pred:
                score.tp += 1
            elif pred:
                score.fp += 1
            elif truth:
                score.fn += 1
            else:
                score.tn += 1
        result.rows.append(detail)
    return result


def _fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


def report(result: EvalResult, verbose: bool = False) -> str:
    lines = [
        f"split: {result.split}   labeled chats: {result.n_chats}"
        + (f"   (goal taken from first user message in {result.goals_from_first_message})"
           if result.goals_from_first_message else ""),
        "",
        f"{'mode':<24}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'?':>4}  precision  recall     F1",
    ]
    for m, s in result.modes.items():
        lines.append(
            f"{m:<24}{s.tp:>4}{s.fp:>4}{s.fn:>4}{s.tn:>4}{s.skipped:>4}"
            f"      {_fmt(s.precision)}   {_fmt(s.recall)}  {_fmt(s.f1)}"
        )
    if verbose:
        lines += ["", "per chat (labeled → predicted):"]
        for r in result.rows:
            mark = "  " if set(r["labeled"]) == set(r["predicted"]) else "✗ "
            lines.append(f"{mark}{r['id']}  {r['labeled'] or ['healthy']} → {r['predicted'] or ['none']}")
    return "\n".join(lines)


def log_test_run(eval_dir: str, result: EvalResult, cfg: Config) -> int:
    """Append this test-set evaluation to test_log.csv; return how many there have been."""
    path = Path(eval_dir) / "test_log.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "config"] + [f"{m}_f1" for m in EVALUATED_MODES])
        w.writerow(
            [datetime.now(timezone.utc).isoformat(timespec="seconds"),
             json.dumps(dataclasses.asdict(cfg))]
            + [_fmt(result.modes[m].f1).strip() for m in EVALUATED_MODES]
        )
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1
