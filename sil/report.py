"""Text report: what happened when the user pushed back, with evidence and one tip per flag."""

from __future__ import annotations

from .loader import Turn
from .pushback import Circling, PushbackEvent

TIPS = {
    "caved": (
        "ask the same question in a new chat, without hinting at an answer, and compare. "
        "an answer that flips when you only push back isn't reliable either way"
    ),
    "circling": (
        "start a new chat with the current code or text, the exact error and what you already "
        "tried. after a few failed rounds models rarely get back on track in the same chat"
    ),
}


def quote(text: str, n: int = 90) -> str:
    s = " ".join(text.split())
    return f'"{s[:n]}..."' if len(s) > n else f'"{s}"'


def pushback_report(turns: list[Turn], events: list[PushbackEvent], circles: list[Circling]) -> str:
    pushes = [e for e in events if e.is_pushback]
    if not pushes:
        return "no pushback found."
    caved = [e for e in pushes if e.outcome == "caved"]
    bare = sum(e.bare for e in pushes)
    lines = [
        f"pushback: {len(pushes)} turn{'s' if len(pushes) != 1 else ''} "
        f"({bare} with no new information), {len(caved)} caved, {len(circles)} circling"
    ]
    for e in caved:
        lines += [
            "",
            f"[CAVED] turn {e.turn}: you pushed back without new information and the AI changed its answer",
            f"    you      (turn {e.turn}): {quote(turns[e.turn].text)}",
            f"    before   (turn {e.turn - 1}): {quote(turns[e.turn - 1].text)}",
            f"    after    (turn {e.turn + 1}): {quote(turns[e.turn + 1].text)}",
            f"    tip: {TIPS['caved']}",
        ]
    for c in circles:
        inside = [e for e in events if c.start_turn <= e.turn <= c.end_turn]
        lines += ["", f"[CIRCLING] turns {c.start_turn}-{c.end_turn}: {c.rounds} pushback rounds in a row"]
        for e in inside[:3]:
            lines.append(f"    turn {e.turn}: {quote(turns[e.turn].text, 70)}")
        if len(inside) > 3:
            lines.append(f"    ... {len(inside) - 3} more")
        lines.append(f"    tip: {TIPS['circling']}")
    held = [e for e in pushes if e.bare and e.outcome == "held"]
    if held:
        lines += ["", "held its answer after bare pushback: turn " + ", ".join(str(e.turn) for e in held)]
    return "\n".join(lines)
