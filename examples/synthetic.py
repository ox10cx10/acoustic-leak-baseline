"""Run the pipeline on simulated recordings; this is not leak-detection evidence."""

import numpy as np

from acoustic_leak import cross_validate, make_pipeline


def make_signals(n_groups=12, clips_per_group=4, seed=7):
    """Generate repeated clips sharing a simulated recording condition."""
    rng = np.random.default_rng(seed)
    time = np.arange(8192) / 8192
    signals, labels, groups = [], [], []
    for group in range(n_groups):
        carrier = rng.uniform(150, 450)
        gain = rng.uniform(0.3, 2.0)
        background = rng.uniform(0.4, 1.0)
        for clip in range(clips_per_group):
            label = clip % 2
            phase = rng.uniform(0, 2 * np.pi)
            modulation = 1 + (0.1 + 0.15 * label) * np.sin(2 * np.pi * 7 * time)
            tone = modulation * np.sin(2 * np.pi * carrier * time + phase)
            high_tone = (0.2 + 0.15 * label + rng.uniform(-0.15, 0.15))
            signal = tone + high_tone * np.sin(2 * np.pi * 1300 * time + phase)
            signal += background * rng.normal(size=time.size)
            signals.append(gain * signal + rng.uniform(-0.2, 0.2))
            labels.append(label)
            groups.append(f"recording-{group:02d}")
    return np.asarray(signals), np.asarray(labels), np.asarray(groups)


def main():
    signals, labels, groups = make_signals()
    result = cross_validate(signals, labels, groups, C=0.1)
    print("Synthetic demonstration only; these scores do not measure real leak detection.")
    print("Rows: true class; columns: predicted class. Class order: [0, 1].")
    print(np.asarray(result["metrics"]["confusion_matrix"]))
    for key in ("accuracy", "leak_recall", "false_positive_rate"):
        print(f"{key}: {result['metrics'][key]:.3f}")

    model = make_pipeline(C=0.1).fit(signals, labels)
    new_signals, _, _ = make_signals(n_groups=1, seed=17)
    score = model.predict_proba(new_signals[:1])[0, 1]
    print(f"New simulated clip: class-1 score = {score:.3f}; label = {int(score >= 0.5)}")


if __name__ == "__main__":
    main()
