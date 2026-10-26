"""Draw the README overview; requires the optional matplotlib dependency."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from examples.synthetic import make_signals


NAVY, TEAL, ORANGE = "#17395C", "#168C92", "#D98734"
GRAY, LIGHT = "#627384", "#DCE4EB"


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": NAVY, "axes.labelcolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(16, 5), dpi=100, facecolor="white")
    fig.text(0.05, 0.93, "Sound to a classification score", fontsize=23, weight="bold")
    for x, title in ((0.05, "1  Prepare the recording"),
                     (0.40, "2  Extract 93 features"),
                     (0.77, "3  Logistic regression")):
        fig.text(x, 0.81, title, fontsize=14, weight="bold")

    signals, _, _ = make_signals(n_groups=1)
    signal = signals[0] - signals[0].mean()
    signal /= np.sqrt(np.mean(signal ** 2))
    wave = fig.add_axes([0.055, 0.37, 0.235, 0.32])
    wave.plot(np.arange(8192) / 8192, signal, color=TEAL, linewidth=0.4)
    wave.set(xlim=(0, 1), xlabel="Time / s", ylabel="Normalized amplitude")
    wave.set_xticks([0, 0.5, 1])
    wave.set_yticks([-3, 0, 3])
    fig.text(0.055, 0.72, "Illustrative signal · 1 s · 8192 Hz", color=GRAY, fontsize=10)
    fig.text(0.055, 0.19, "Mean removal + RMS normalization", fontsize=11)
    fig.text(0.055, 0.13, "Applied independently to each recording", color=GRAY, fontsize=10)

    bars = fig.add_axes([0.455, 0.405, 0.19, 0.25])
    bars.barh([2, 1, 0], [15, 64, 14], height=0.43,
              color=[NAVY, TEAL, ORANGE])
    bars.set(xlim=(0, 73), ylim=(-0.5, 2.5))
    bars.set_yticks([2, 1, 0], ["Global", "Log-Mel", "Temporal"])
    bars.tick_params(axis="y", length=0, labelcolor=NAVY)
    bars.set_xticks([])
    for y, count in zip([2, 1, 0], [15, 64, 14]):
        bars.text(count + 2, y, str(count), va="center", fontsize=12, weight="bold")
    for spine in bars.spines.values():
        spine.set_visible(False)
    fig.text(0.40, 0.19, "15 + 64 + 14 = 93 descriptors", fontsize=11)
    fig.text(0.40, 0.13, "Feature count by block", color=GRAY, fontsize=10)

    fig.text(0.79, 0.72, r"$d = \mathbf{w}^{\mathsf{T}}\mathbf{z} + b$", fontsize=17)
    sigmoid = fig.add_axes([0.79, 0.37, 0.17, 0.29])
    margin = np.linspace(-6, 6, 301)
    sigmoid.plot(margin, 1 / (1 + np.exp(-margin)), color=NAVY, linewidth=2.4)
    sigmoid.axhline(0.5, color=ORANGE, linewidth=1, linestyle="--")
    sigmoid.axvline(0, color=LIGHT, linewidth=1, zorder=0)
    sigmoid.plot([0], [0.5], "o", color=ORANGE, markersize=5)
    sigmoid.set(xlim=(-6, 6), ylim=(-0.04, 1.04), xlabel="Decision score d", ylabel="Score s")
    sigmoid.set_xticks([-6, 0, 6])
    sigmoid.set_yticks([0, 0.5, 1])
    fig.text(0.77, 0.19, r"$s = 1/(1+e^{-d})$", fontsize=13)
    fig.text(0.77, 0.13, "s ≥ 0.5: leak    s < 0.5: non-leak", fontsize=10)

    for left, right, label in ((0.315, 0.37, "Feature\nextraction"),
                                (0.68, 0.75, "Feature\nstandardization")):
        arrow = FancyArrowPatch((left, 0.53), (right, 0.53),
                                transform=fig.transFigure, arrowstyle="-|>",
                                mutation_scale=15, color=GRAY, linewidth=1.4)
        fig.add_artist(arrow)
        fig.text((left + right) / 2, 0.415, label, ha="center", va="top",
                 fontsize=10, color=GRAY, linespacing=1.4)
    for ax in (wave, sigmoid):
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(LIGHT)

    destination = Path(__file__).resolve().parents[1] / "assets"
    destination.mkdir(exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(destination / f"overview.{extension}", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
