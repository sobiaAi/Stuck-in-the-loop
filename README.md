# stuck-in-the-loop

Finds the places where a chat with an AI goes wrong after you push back.

When you tell an AI "are you sure?" or "that's wrong", it should change its
answer only if you gave it a reason to. Two things go wrong instead:

1. **Caving**: you pushed back without any new information and the AI
   changed its answer anyway. Now you can't trust either answer.
2. **Circling**: you push back on the same problem again and again ("still
   not working", "same error") and the chat goes nowhere.

In real ChatGPT conversations ([WildChat](https://huggingface.co/datasets/allenai/WildChat-1M)),
after a bare "are you sure?" / "that's wrong" / "No.", the AI changed its
answer **58% of the time** (72 of 124 labeled cases, 95% CI 49-66%). That's
close to what FlipFlop (Laban et al., 2023) measured in controlled
experiments (46%), but here it's real users.

`sil` finds these moments in any chat export. No LLM is involved: a small
sentence embedding model, a small NLI model and two logistic regressions.

## Install

```bash
git clone https://github.com/sobiaAi/stuck-in-the-loop
cd stuck-in-the-loop
pip install -e .
```

The first run downloads all-MiniLM-L6-v2 (90 MB) and nli-deberta-v3-xsmall
(290 MB).

## Usage

```bash
python -m sil analyze chat.json
```

```
pushback: 1 turn (1 with no new information), 1 caved, 0 circling

[CAVED] turn 2: you pushed back without new information and the AI changed its answer
    you      (turn 2): "Are you sure?"
    before   (turn 1): "The three founders of Finxter are Christian Mayer, Lukas Rieger, and Hans Meier."
    after    (turn 3): "I apologize for the confusion in my previous response. The main person behind Finxter is C..."
    tip: ask the same question in a new chat, without hinting at an answer, and compare. an answer that flips when you only push back isn't reliable either way
```

It also saves a chart (`--out chart.png`) with the pushback probability of
every user turn on top.

## How it works

Every user turn that answers an AI turn is an event. Two questions decide
what happened:

|  | AI changed its answer | AI kept its answer |
|---|---|---|
| **you gave no new information** | caving | held (good) |
| **you gave new information** (an error, a reason, a correction) | updated (good) | kept; if it repeats, circling |

- **Pushback?** Logistic regression on 17 cheap features of the user turn
  (negations, doubt words, "still", error logs, overlap with the AI's
  answer, whether the AI had asked a question) plus the turn's MiniLM
  embedding.
- **Bare?** What's left of the user turn after removing the challenge
  ("are you sure", "that's wrong", "try again"). If nothing of its own is
  left (no error, reason, number or new content word), it's bare.
- **Changed?** Compares the AI's answers before and after. Embeddings can't
  do this ("the total is 198" and "the total is 194" are almost the same
  vector), so it uses an NLI model: does the new first claim or conclusion
  contradict the old one? Plus a few cheap signals (repeated text, "Yes, I'm
  sure", asking for clarification, changed numbers or options).
- **Caving** = pushback + bare + changed, unless the old answer was only a
  hedged guess ("it might be...").
- **Circling** = at least 3 user turns in a row that push back.

## Results

Models were fit on WildChat shard 0 and frozen. Test sets come from shard 1,
were labeled before the frozen models ever saw them, and were run once.
Details in [`eval/PUSHBACK.md`](eval/PUSHBACK.md).

**Events** (300 test events, 140 pushback, 24 caving)

| | precision | recall | F1 |
|---|---|---|---|
| pushback | 0.82 | 0.86 | 0.84 |
| bare (given pushback) | 0.90 | 0.66 | 0.76 |
| changed (given bare pushback) | 0.76 | 0.81 | 0.78 |
| **caving** | 0.55 | 0.67 | 0.60 |

**Circling** (130 test chats, 38 circling)

| | precision | recall | F1 |
|---|---|---|---|
| **sil** | 0.59 | 0.79 | 0.67 |
| 3 user turns in a row with cue words | 0.55 | 0.71 | 0.62 |
| low novelty (v0.1 of this tool) | 0.48 | 0.29 | 0.36 |

The first version judged "changed" from how the reply opened ("I apologize,
the correct answer is..."). Its caving detection scored F1 0.88 on the data
it was built on and 0.22 on its blind test: it caught 1 of 12 real answer
changes, because ChatGPT apologizes whether or not it changes anything. Comparing the two answers fixed most of that. On an outside
benchmark it never saw, SYCON-Bench (Hong et al., 2025), the answer-change
check reaches F1 0.54 against 0.10 without the NLI features.

## What the labels show

From 124 labeled bare pushbacks:

- the AI changed its answer in 58% [49, 66] (53% on shard 0, 64% on shard 1)
- in 40% [31, 48] it was a real claim that flipped (not code, stories, or
  things only the user knows)
- in 19% [13, 27] it apologized and gave the same answer again
- GPT-4 did not hold its ground better than GPT-3.5 (78% of 18 vs 55% of
  106 changed), though the GPT-4 sample is small

With new information (an error, a reason), the AI changed its answer in 86%.

## Input formats

| Format | How to get it |
|---|---|
| ChatGPT export | Settings > Data controls > Export data, then use `conversations.json` from the zip |
| Claude export | Settings > Privacy > Export data, then use `conversations.json` |
| Pasted text (`.txt` / `.md`) | copy a chat into a file. each message starts with `You said:` / `ChatGPT said:`, `User:` / `Assistant:`, or `## User` / `## Assistant` |
| json | `{"session_id": "...", "turns": [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}]}` |

Export files have all your chats in them. List them and pick one by number or
part of the title:

```bash
python -m sil list conversations.json
python -m sil analyze conversations.json --conversation "study plan"
```

## Other signals

The chart also shows three per-turn signals from the first version of the
tool: novelty (how different a turn is from the last three), drift from the
goal (`--goal`, default the first user message) and agreement phrases. They
flag validation loops, premature convergence and drift on made-up example
chats, but these patterns turned out to be rare or not visible in real
chats, and they haven't been validated on real ones. Treat them as context.

## Limitations

- about half of the caving flags are wrong. most false alarms are an apology
  around the same answer, or code being rewritten after "it's wrong, try again"
- the bare check misses pushback that names the item without a reason
  ("the founder is wrong"), and "double check please" is often not seen as
  pushback
- English only, tested on ChatGPT (GPT-3.5 / GPT-4) chats from 2023
- the labels were made with an LLM following the written guides in `eval/`.
  the test sets were labeled without access to the detector. each test item was labeled twice
  independently (agreement: pushback κ 0.96, caving κ 0.93, circling κ 0.94)
  and disagreements went to a third label. a hand-checked sample is in progress
- works on finished chats, not live

## License

MIT
