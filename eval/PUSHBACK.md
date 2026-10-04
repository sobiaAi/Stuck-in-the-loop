# Pushback evaluation

How the pushback, caving and circling detectors were built and tested.

## Definitions

An **event** is one user turn that answers an AI turn, plus the AI's next
reply.

| Label | Meaning |
|---|---|
| `pushback` | the user objects to the AI's last answer: says it's wrong, doubts it, says it doesn't work or is still not what they asked |
| `new_info` | the pushback brings something of its own: an error, what they saw when they ran it, a reason, a counter-claim, a correction of what they meant. `0` = bare ("are you sure?", "that's wrong", "No.") |
| `ai_changed` | the AI's next reply gives a different answer (not just an apology around the same answer) |
| `capitulation` | bare pushback and the AI changed a claim of fact or reasoning. Not counted: code iteration ("wrong, try again" on a program), creative or document redo, the user being the authority (what they meant, which song they remember, a guess the AI hedged), advice |

**Circling** (chat level): at least 3 user pushback turns in a row about the
same unresolved problem.

Full labeling guides: [`event_guide.md`](event_guide.md),
[`circling_guide.md`](circling_guide.md).

## Data

[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (Zhao et
al., 2024, ODC-BY). English, non-toxic, at least 3 user turns, at least 90%
Latin letters on the user side, one chat per distinct opening message.

- **dev**: `train-00000-of-00014.parquet` (6,945 chats, the 60 chats in
  `eval/data/` left out)
- **test**: `train-00001-of-00014.parquet` (6,362 chats, openings that also
  appear in shard 0 left out)

Events were sampled one per chat, stratified by cue words in the user turn
(short bare challenges / strong cues / weak cues only), so pushback is much
more common in the samples than in real chats. Circling chats were sampled by
the longest run of cue words (3+, 2, 1, 0).

| Set | Shard | Size | Used for |
|---|---|---|---|
| D1, T1 | 0 | 240 + 240 events | building v1 and v2 |
| T2 | 0 | 200 events | blind test of v1 (run once), then building v2 |
| T3 | 1 | 300 events | blind test of v2 (run once) |
| circling dev | 0 | 130 chats | choosing the circling rule |
| circling test | 1 | 130 chats | blind test of circling (run once) |

## Protocol

- models are fit on dev sets only and frozen (`sil/weights.json`) before the
  test labels exist
- every run on a test set is logged; each test set was run once
- the test sets were labeled twice independently; disagreements went to a
  third label and the majority was kept
- the two clarifications at the end of the guides' definitions (hedged guesses,
  re-sending a request) were added before labeling T3 and the circling test set;
  the dev sets were labeled without them

## Results

T2, v1 (hand-written rules for "changed"), run once:

| | P | R | F1 |
|---|---|---|---|
| pushback | 0.74 | 0.79 | 0.76 |
| bare (given pushback) | 0.92 | 0.82 | 0.87 |
| changed (given bare) | 0.25 | 0.08 | 0.12 |
| caving | 0.33 | 0.17 | 0.22 |

T3, v2 (frozen `sil/weights.json`), run once. 95% Wilson intervals:

| | P | R | F1 |
|---|---|---|---|
| pushback | 0.82 [0.75, 0.87] | 0.86 [0.79, 0.91] | 0.84 |
| bare (given pushback) | 0.90 [0.77, 0.96] | 0.66 [0.53, 0.77] | 0.76 |
| changed (given pushback) | 0.81 [0.72, 0.88] | 0.71 [0.61, 0.79] | 0.76 |
| changed (given bare) | 0.76 [0.61, 0.87] | 0.81 [0.65, 0.90] | 0.78 |
| caving | 0.55 [0.38, 0.72] | 0.67 [0.47, 0.82] | 0.60 |

Circling test (130 chats, 38 circling), run once:

| | P | R | F1 | span IoU |
|---|---|---|---|---|
| sil (3+ pushback turns in a row, p >= 0.4) | 0.59 | 0.79 | 0.67 | 0.78 |
| 3 user turns in a row with cue words | 0.55 | 0.71 | 0.62 | 0.77 |
| low novelty (v0.1) | 0.48 | 0.29 | 0.36 | 0.48 |

On the circling dev set, adding a recurrence check (laminarity of the AI-turn
recurrence matrix) or a same-topic check raised precision but lowered F1, so
neither is used.

External check for "changed": SYCON-Bench debate setting (Hong et al., 2025),
prompt 0, GPT-4o / Llama-3.1-8B / Qwen2.5-7B, 901 consecutive answer pairs
(243 stance flips by the benchmark's own judge). Fit on WildChat only:

| | P | R | F1 |
|---|---|---|---|
| v1 rules | 0.33 | 0.00 | 0.01 |
| v2 without NLI features | 0.29 | 0.06 | 0.10 |
| v2 (used) | 0.74 | 0.42 | 0.54 |

Labeler agreement on the test sets (two independent labels): pushback
κ 0.96, bare κ 0.87, changed κ 0.94, caving κ 0.93, circling κ 0.94.
