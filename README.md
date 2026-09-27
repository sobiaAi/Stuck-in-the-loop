# stuck-in-the-loop

Human-AI conversations have trajectories. Some converge on useful output; others
decay into recognizable failure patterns without either party noticing:

1. **Validation loop** — the user seeks agreement, the AI supplies it, and
   information content flatlines. Turns keep happening; nothing new enters the
   session.
2. **Premature convergence** — both parties lock onto an early, mediocre
   solution and stop exploring the space.
3. **Drift** — the session wanders progressively further from its stated goal
   without converging on anything.

stuck-in-the-loop treats a session as a discrete dynamical system: each turn
updates a small state vector of measurable signals — novelty, drift from goal,
agreement density — and failure modes appear as characteristic patterns in the
state trajectory. v0.1 analyzes transcripts offline and renders a trajectory
chart with flagged regions.

## Install

```bash
git clone https://github.com/sobiaAi/stuck-in-the-loop
cd stuck-in-the-loop
pip install -e .
```

## Usage

```bash
python -m sil analyze examples/validation_loop.json --out chart.png
```

Output: a trajectory chart (three stacked series — novelty, drift, agreement —
with flagged regions shaded) plus a text summary:

```
[VALIDATION LOOP] turns 4–17, severity 0.78
chart saved to chart.png
```

![Trajectory chart of the validation-loop example](docs/validation_loop.png)

Results on the four example sessions in `examples/`:

| Session | Verdict |
|---|---|
| validation loop | `[VALIDATION LOOP] turns 4–17, severity 0.78` |
| premature convergence | `[PREMATURE CONVERGENCE] turns 10–17, severity 0.44` |
| drift | `[DRIFT] turns 11–19, severity 0.45` |
| healthy (control) | no failure modes detected |

Charts for all four are in [`docs/`](docs/).

## Input formats

`CHAT` can be any of:

| Format | How to get it |
|---|---|
| **ChatGPT export** | Settings → Data controls → Export data; use `conversations.json` from the zip |
| **Claude export** | Settings → Privacy → Export data; use `conversations.json` |
| **Pasted text** (`.txt` / `.md`) | Copy a chat and save it. Messages start with a speaker label: `You said:` / `ChatGPT said:`, `User:` / `Assistant:`, or `## User` / `## Assistant` |
| **stuck-in-the-loop JSON** | See below |

Export files hold every conversation you've had. List them, then pick one by
index or by part of its title:

```bash
python -m sil list conversations.json
python -m sil analyze conversations.json --conversation "study plan" --goal "Build a two-week study plan"
```

Drift is measured against the session's goal. Give it with `--goal`, or in
the `goal` field of the JSON format.

```json
{
  "session_id": "my-session",
  "goal": "optional: one-sentence statement of what the session is trying to achieve",
  "turns": [
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ]
}
```

## How detection works

Each turn gets three signals:

- **Novelty** — `1 − max cosine similarity` to the previous 3 turns, clamped
  to [0, 1]. Max rather than mean: a turn that repeats *any* recent turn is
  repetitive, even if it differs from the others.
- **Drift** — `1 − cosine similarity` to the session goal, smoothed with an
  exponential moving average (α = 0.3) so a single tangent doesn't count;
  only a sustained move away does.
- **Agreement density** — agreement markers per word, counted separately for
  agreement-*seeking* user turns ("right?", "am I correct") and
  agreement-*supplying* assistant turns ("exactly", "you're right"), then
  averaged over the last 4 turns. A validation loop needs both sides.

Raw novelty alternates turn to turn, because each turn is also compared with
the other speaker's message, so the rules use its average over the last 4
turns.

| Failure mode | Rule (for at least 4 consecutive turns) |
|---|---|
| Validation loop | agreement > 0.03 and smoothed novelty < 0.45 |
| Drift | drift > 0.85, and it doesn't come back down |
| Premature convergence | smoothed novelty < 0.30 through the end of the session, with agreement ≤ 0.03 |

Severity is the fraction of the session the flagged span covers. Thresholds
live in the `Config` dataclass (override any of them with `--config
thresholds.json`) and were calibrated on the synthetic sessions in
`examples/`:

| Setting | Value |
|---|---|
| `novelty_window` | 3 |
| `agreement_window` | 4 |
| `drift_alpha` | 0.3 |
| `min_run` | 4 |
| `agreement_ceiling` | 0.03 |
| `loop_novelty_max` | 0.45 |
| `novelty_floor` | 0.30 |
| `drift_ceiling` | 0.85 |

## Evaluation on real conversations

The thresholds above were set on four synthetic sessions, which is not
evidence they generalize. Evaluation uses real conversations from
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), labeled by
hand under a written protocol ([`eval/LABELING.md`](eval/LABELING.md)):

```bash
pip install -e .[eval]
python -m sil sample-wildchat train-00000-of-00014.parquet --n 60 --out eval
# label eval/labels.csv by reading eval/data/<id>.md — before running the detector
python -m sil evaluate eval                  # calibration half: tune freely
python -m sil evaluate eval --split test     # held-out half: every run is logged
```

The calibration/test split is fixed when the sample is drawn, before anything
is labeled, and every evaluation on the test half is recorded in
`eval/test_log.csv`.

`--recurrence plot.png` on `analyze` also saves the conversation's recurrence
plot (cosine similarity between every pair of turns).

## Limitations (v0.1)

- Agreement markers are lexical, heuristic, and English-only.
- Thresholds are hand-calibrated on four synthetic sessions, not learned —
  evaluation on real conversations is the next step.
- Drift measures topical distance from the goal, not usefulness: a turn that
  serves the goal indirectly (e.g. setting an alarm to study earlier) still
  reads as drift.
- The goal must be supplied and is trusted as given; inferring it from early
  turns, and telling a deliberate change of goal apart from drift, is future work.
- Offline analysis only — real-time monitoring is planned for v0.2.

## License

MIT
