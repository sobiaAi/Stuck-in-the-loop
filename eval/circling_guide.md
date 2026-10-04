# Chat labeling guide: circling

Each item is a whole real conversation between a user and ChatGPT (WildChat,
2023). Turns are numbered [0], [1], ... Long turns are clipped with "[...]".
Chats longer than 40 turns are cut off; judge only what is shown.

## Definition

**Pushback turn**: a user turn that objects to the AI's previous reply: says
it is wrong, doubts it, says it still doesn't work / gives the same error /
the same output, or says it is still not what they asked for. New requests,
modification requests ("now make it blue"), follow-up questions and answers
to the AI's questions are NOT pushback.

**Circling** = the conversation goes in circles on ONE problem: there are at
least 3 pushback turns in a row (consecutive user turns, nothing else in
between) about the same unresolved problem. The AI replies in between
either repeat themselves (often after an apology) or keep changing things
without fixing it. A new, different request breaks the run. Sending the same request again after
a refusal or a failure, without new requirements, counts as pushback.

Not circling:
- fewer than 3 consecutive pushbacks on the problem (2 failed rounds then fixed is not circling)
- the user iterating on requirements where each round works and adds something
- several different problems, each fixed within 1-2 rounds
- a roleplay/story where "no" or "again" is part of the game

## Columns

`k,circling,start,end,repeat,note`

- `circling`: 1 or 0
- `start`, `end`: if circling = 1, the turn numbers of the FIRST and LAST
  user pushback turn of the longest circling run (e.g. 6 and 12). Empty if 0.
- `repeat`: if circling = 1, 1 when at least once in the run the AI gave
  essentially the same answer/code again (apology-and-repeat), else 0. Empty if 0.
- `note`: a few words, e.g. "4x 'still same error' on flask import; AI rewrites
  code each time" or "2 pushbacks then fixed".

One row per chat, in order of k.
