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
[VALIDATION LOOP] turns 8–14, severity 0.81
chart saved to chart.png
```

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

| Failure mode | Signal signature |
|---|---|
| Validation loop | novelty flatlines below floor while agreement density stays high on both sides |
| Premature convergence | novelty drops early and stays down, drift stays low — converged, just too early |
| Drift | drift-from-goal series trends upward without recovering |

Novelty is `1 − max cosine similarity` to the last few turns; drift is
`1 − cosine similarity` to the declared goal, EMA-smoothed; agreement density
is a lexical marker score computed separately for agreement-*seeking* (user)
and agreement-*supplying* (assistant) turns, since a validation loop requires
both sides.

Thresholds live in a `Config` dataclass and were calibrated on the synthetic
sessions in `examples/`.

## Limitations (v0.1)

- Agreement markers are lexical, heuristic, and English-only.
- Thresholds are hand-calibrated on synthetic sessions, not learned.
- The goal must be supplied and is trusted as given; inferring it from early
  turns, and telling a deliberate change of goal apart from drift, is future work.
- Offline analysis only — real-time monitoring is planned for v0.2.

## License

MIT
