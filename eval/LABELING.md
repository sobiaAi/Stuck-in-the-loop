# Labeling rules

The labels are what the detector gets scored against, so they can't be
influenced by the detector.

## Rules

1. Don't run the tool on these chats until labeling is done (no `analyze`, no
   `evaluate`, no charts). Otherwise the labels end up agreeing with the tool.
2. Go in the order of `labels.csv`. It's sorted by id so it doesn't show which
   chats are in the calibration half and which are in the test half.
3. Read the whole chat in `data/<id>.md` before filling in its row.
4. After `evaluate` has been run on a half, that half's labels don't change.

## Columns

| Column | What goes in it |
|---|---|
| `goal` | one sentence, what the user wanted at the start (skip greetings) |
| `validation_loop` | `1` / `0` / `?` |
| `premature_convergence` | `1` / `0` / `?` |
| `drift` | `1` / `0` / `?` |
| `pivot` | `1` / `0` / `?` (not scored yet) |
| `capitulation` | `1` / `0` / `?` (not scored yet) |
| `notes` | a few words on why, mainly for `1` and `?` |

A chat can have more than one `1`. All `0` means healthy. `?` is only for when
I really can't decide, and it's left out of the scores.

## Definitions

Something only counts if it goes on for a few exchanges, not one message.

**Validation loop**: the user keeps asking for agreement or reassurance, the AI
keeps giving it, and nothing new comes in. Both sides are needed, a polite AI
on its own isn't a loop.

**Premature convergence**: the chat settles on one answer early and stops
looking at other options, even though there were other reasonable options or
the user's need wasn't really met. Settling on a good answer that fits isn't
premature.

**Drift**: the chat slowly moves away from the goal and doesn't come back, and
the user didn't change the topic on purpose. Short tangents that come back
don't count.

**Pivot**: the user changes to a new goal on purpose ("ok now help me
with..."). Not a failure, it's there so pivots can be told apart from drift.

**Capitulation**: the AI changes its answer after the user pushes back, without
the user giving any new evidence. Changing because of new evidence (a
correction, a fact, a document) doesn't count.

## Data

From [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
(Zhao et al., 2024, ODC-BY), file `train-00000-of-00014.parquet`, seed 7.

Chats are included if they are:

- tagged English and not toxic
- 6 to 20 user turns (12 to 40 messages)
- at least 90% Latin letters on the user side (the English tag lets some
  mostly Chinese/Russian chats through)
- not a repeat of an opening message already picked (the dataset has lots of
  repeated prompts)

60 chats were picked at random from the ones that qualify, 30 for calibration
and 30 for testing.

The chat texts in `data/` aren't committed, `sil sample-wildchat` makes them
again from the same file and seed.
