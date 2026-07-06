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

<!-- TODO: embed example chart screenshot here once metrics are implemented -->

## Transcript format

```json
{
  "session_id": "my-session",
  "goal": "one-sentence statement of what the session is trying to achieve",
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

<!-- TODO: record final tuned threshold values here -->

## Limitations (v0.1)

- Agreement markers are lexical, heuristic, and English-only.
- Thresholds are hand-calibrated on synthetic sessions, not learned.
- The `goal` field is trusted as given; goal inference from early turns is
  future work.
- Offline analysis only — real-time monitoring is planned for v0.2.

## License

MIT
