"""Trajectory chart rendering and text summary."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .loader import Transcript
from .metrics import Flag

FLAG_COLORS = {
    "validation_loop": "#d62728",  # red
    "premature_convergence": "#ff7f0e",  # orange
    "drift": "#9467bd",  # purple
}

FLAG_LABELS = {
    "validation_loop": "Validation loop",
    "premature_convergence": "Premature convergence",
    "drift": "Drift",
}


def render(
    transcript: Transcript,
    novelty_s: list[float],
    drift_s: list[float],
    agreement_s: list[float],
    flags: list[Flag],
    out_path: str,
) -> None:
    """Save a PNG: three stacked line charts (novelty, drift, agreement)
    sharing the turn-index x-axis, with flagged regions shaded."""
    n = len(transcript.turns)
    x = range(n)

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(14, 8), dpi=110, constrained_layout=True
    )

    # agreement is markers per word, so it lives near 0.0–0.1; a 0–1 axis flattens it
    agreement_top = max(0.1, 1.25 * max(agreement_s, default=0.0))
    series = [
        (novelty_s, "Novelty", "#1f77b4", 1.0),
        (drift_s, "Drift from goal", "#2ca02c", 1.0),
        (agreement_s, "Agreement density", "#7f7f7f", agreement_top),
    ]

    for ax, (values, label, color, top) in zip(axes, series):
        ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=3.5)
        ax.set_ylabel(label, fontsize=11)
        ax.set_ylim(-0.05 * top, 1.05 * top)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for flag in flags:
            ax.axvspan(
                flag.start_turn - 0.5,
                flag.end_turn + 0.5,
                color=FLAG_COLORS.get(flag.mode, "#999999"),
                alpha=0.18,
                linewidth=0,
            )

    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("Turn index", fontsize=11)
    axes[-1].set_xlim(-0.5, n - 0.5)

    seen_modes = sorted({f.mode for f in flags})
    if seen_modes:
        handles = [
            Patch(
                facecolor=FLAG_COLORS.get(m, "#999999"),
                alpha=0.35,
                label=FLAG_LABELS.get(m, m),
            )
            for m in seen_modes
        ]
        axes[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=10)

    fig.suptitle(transcript.session_id, fontsize=14, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def render_recurrence(transcript: Transcript, E, out_path: str) -> None:
    """Save a recurrence plot: cell (i, j) is the cosine similarity of turns i and j."""
    import numpy as np

    S = np.clip(E @ E.T, 0.0, 1.0)
    n = len(S)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=110, constrained_layout=True)
    im = ax.imshow(S, cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xlabel("Turn index", fontsize=11)
    ax.set_ylabel("Turn index", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    roles = "".join("U" if t.role == "user" else "A" for t in transcript.turns)
    if n <= 40:
        ax.set_xticks(range(n), [f"{i}{roles[i]}" for i in range(n)], fontsize=7, rotation=90)
        ax.set_yticks(range(n), [f"{i}{roles[i]}" for i in range(n)], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    ax.set_title(f"{transcript.session_id} — recurrence plot", fontsize=13, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def summarize(flags: list[Flag]) -> str:
    """One line per flag: [VALIDATION LOOP] turns 8-14, severity 0.81."""
    if not flags:
        return "No failure modes detected."
    lines = []
    for f in flags:
        mode = f.mode.replace("_", " ").upper()
        lines.append(
            f"[{mode}] turns {f.start_turn}–{f.end_turn}, severity {f.severity:.2f}"
        )
    return "\n".join(lines)
