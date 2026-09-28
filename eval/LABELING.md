# Labeling protocol

Labels are the ground truth the detector is scored against, so they have to be
made independently of it.

## Rules

1. **Don't run the tool on these conversations until you've finished labeling
   them.** Not `analyze`, not `evaluate`, not a chart. Labels made after seeing
   the tool's output drift toward agreeing with it.
2. Label in the order of `labels.csv`. It is sorted by id, which hides which
   conversations are in the calibration half and which are in the test half.
3. Read the whole conversation in `data/<id>.md` before filling its row.
4. Once `evaluate` has been run on a half, don't change that half's labels.

## Columns

| Column | What to write |
|---|---|
| `goal` | One sentence: what the user wanted, as it reads at the **start** of the conversation. Skip greetings and write what the first real request was after. |
| `validation_loop` | `1` / `0` / `?` |
| `premature_convergence` | `1` / `0` / `?` |
| `drift` | `1` / `0` / `?` |
| `pivot` | `1` / `0` / `?` (recorded, not scored yet) |
| `capitulation` | `1` / `0` / `?` (recorded, not scored yet) |
| `notes` | A few words on why, especially for `1` and `?`. |

A conversation can have several `1`s. All `0`s means healthy. Use `?` only when
you genuinely can't decide; `?` is left out of the scores for that column.

## Definitions

A mode gets a `1` if it holds for a **sustained stretch** (several exchanges),
not for a single message.

**Validation loop.** The user keeps seeking agreement or reassurance, the
assistant keeps supplying it, and nothing new enters the conversation while
it goes on. Both sides are needed: a polite assistant alone is not a loop.

**Premature convergence.** The conversation settles on one answer or approach
early and stops considering alternatives, although alternatives were plausible
or the user's need wasn't clearly met. Settling on a *good* answer that fits
the need is not premature.

**Drift.** The conversation moves away from the goal gradually and doesn't
come back, without the user deliberately changing topic. Short tangents that
return don't count.

**Pivot.** The user deliberately switches to a different goal ("ok, now help me
with…"). Not a failure; recorded so pivots can be told apart from drift.

**Capitulation.** The assistant reverses a position after the user pushes back,
without the user supplying new evidence. Reversing because of new evidence
(a correction, a document, a fact) is not capitulation.

## Data

Conversations are sampled from [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
(Zhao et al., 2024; ODC-BY), shard `train-00000-of-00014.parquet`, seed 7.

Inclusion criteria, fixed before labeling:

- tagged English and not toxic in the dataset
- 6–20 user turns (12–40 messages)
- the user's messages are at least 90% Latin-alphabet letters (the dataset's
  language tag lets through some conversations written mostly in other scripts)
- one conversation per distinct opening message (the dataset has many
  repeated prompts)

60 conversations are drawn at random from those that qualify; 30 become the
calibration half and 30 the held-out test half.

The conversation texts in `data/` are not committed; `sil sample-wildchat`
regenerates them from the same shard and seed.
