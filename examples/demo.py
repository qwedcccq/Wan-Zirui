"""End-to-end demo on synthetic signals.

This file contains **placeholders only**:

- two no-op actions that return the input unchanged (they are not methods);
- a detector with illustrative thresholds that differ from the paper's values;
- a minimal illustrative knowledge graph and an empty memory table;
- a scripted prescriber that returns a fixed candidate;
- a do-no-harm callback that accepts everything (demo only).

It shows how the pieces connect.  It is not a reproduction of the paper and
carries none of the paper's values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework import (  # noqa: E402
    ActionRegistry,
    CalibrationMemory,
    DecisionPipeline,
    DictKnowledgeGraph,
    GateController,
    RuleDetector,
    TemplatePrescriber,
    band_ratio_fingerprint,
)

# Illustrative bands; deliberately different from any paper configuration.
LINE_HZ = 60.0
LINE_BAND = 2.0
EMG_BAND = (35.0, 90.0)
REF_BAND = (6.0, 25.0)


def synthetic_record(fs: float = 250.0, seconds: float = 4.0, seed: int = 0):
    """Three channels: background, one 60 Hz tone, one drifting channel."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = rng.normal(0.0, 1.0, size=(3, n))
    x[0] += 0.8 * np.sin(2 * np.pi * LINE_HZ * t)   # line-noise-like channel
    x[1] += 0.02 * t                                 # drift-like channel
    return x


def build_demo():
    fs = 250.0

    # --- toy actions: no-ops, not methods --------------------------------
    registry = ActionRegistry()
    registry.register("placeholder_a", lambda rec, fs_: rec.copy())
    registry.register("placeholder_b", lambda rec, fs_: rec.copy())

    # --- fingerprint: illustrative bands (see module constants) ----------
    def fingerprinter(record, fs_):
        return band_ratio_fingerprint(
            record, fs_,
            line_hz=LINE_HZ, line_band=LINE_BAND,
            emg_band=EMG_BAND, ref_band=REF_BAND,
        )

    # --- detector: illustrative thresholds -------------------------------
    detector = RuleDetector({
        "line_noise": lambda fp: fp[0] > 1.5,
        "channel_corruption": lambda fp: fp[2] > 4.5,
    })

    # --- knowledge graph: minimal illustrative edges ---------------------
    kg = DictKnowledgeGraph({
        "line_noise": [("placeholder_b", 1.0), ("placeholder_a", 0.5)],
        "normal_unknown": [("placeholder_a", 1.0)],
    })

    # --- memory: empty table with illustrative thresholds ----------------
    memory = CalibrationMemory(
        clip_lo=-0.5, clip_hi=1.5,
        min_subjects=2, strong_gain=0.10, separable_gain=0.05,
    )

    # --- prescriber: scripted, no model call -----------------------------
    prescriber = TemplatePrescriber(
        model=lambda prompt: "placeholder_b",
        role="demo",
        schema='{"action": "<candidate id>"}',
    )

    gates = GateController(strong_gain=0.10)
    return fs, registry, fingerprinter, detector, kg, memory, prescriber, gates


def calibrate(pipeline, records, score_fn, subject, memory, fs,
              epochs=None, labels=None):
    """Calibration loop: decide, execute, score, write paired gains back.

    ``score_fn(record, subject) -> float`` is the reader's judge.  This is a
    **single-pass scaffold**, not the paired-fold protocol of the paper: wrap
    it in your own fold/repeat loop.  Feature-stage decisions need ``epochs``
    and ``labels``; no-processing decisions execute as keep-as-is.
    """
    written = []
    for record in records:
        result = pipeline.decide(record, fs)
        spec = (pipeline.registry.get(result.decision.action)
                if result.decision.action != "none" else None)
        baseline = score_fn(record, subject)
        if spec is not None and spec.stage == "feature":
            feature_score = pipeline.execute(record, fs, result.decision,
                                             epochs=epochs, labels=labels)
            gain = (feature_score - baseline) * 100.0
        else:
            processed = pipeline.execute(record, fs, result.decision)
            gain = (score_fn(processed, subject) - baseline) * 100.0
        for key in result.state_key.split("|"):
            memory.write_back(pipeline.context, key, result.decision.action,
                              subject, gain, source="calibration")
        written.append((result.state_key, result.decision.action, gain))
    return written


def main() -> int:
    fs, registry, fingerprinter, detector, kg, memory, prescriber, gates = build_demo()
    pipeline = DecisionPipeline(
        fingerprinter=fingerprinter,
        detector=detector,
        knowledge_graph=kg,
        memory=memory,
        prescriber=prescriber,
        gates=gates,
        registry=registry,
        context="demo|3|synthetic",
        do_no_harm=lambda action: True,   # demo only
        default_action="placeholder_a",
    )

    print("synthetic demo; not a paper result")
    record = synthetic_record(fs)
    result = pipeline.decide(record, fs, materials={"note": "synthetic"})
    print("state:", result.state_key)
    print("action:", result.decision.action)
    print("source:", result.decision.source.value)
    print("strong:", result.decision.strong, "reused:", result.decision.reused)
    print("log:", result.log)

    out = pipeline.execute(record, fs, result.decision)
    print("executed shape:", out.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
