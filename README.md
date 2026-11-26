# LLM-assisted EEG preprocessing harness

A reference implementation of the state-conditioned preprocessing decision
described in the paper. A record is compressed into a fingerprint, mapped to a
state set, and an action is selected from a finite candidate set by layered
evidence — knowledge graph, calibration memory, and an optional language model —
with deterministic gates and paired-measurement evaluation. The harness runs
end to end on your methods, your graph, and your decoder.

## Package layout

| Module | What it does |
|---|---|
| `framework/state_set.py` | Eight state names, set semantics, and a rule-based detector |
| `framework/fingerprint.py` | Fingerprint interface, band-ratio example, concatenation helper |
| `framework/knowledge_graph.py` | Weighted state→action edges, mechanistic ranking, multi-state candidates |
| `framework/memory.py` | Per-subject gain table, Eq. 15 aggregation, sample-size and inseparable-gain rules, parallel contexts, provenance |
| `framework/gates.py` | Pre-gate (strong evidence) and post-gate (in-domain, do-no-harm, fallback) |
| `framework/prescription.py` | Prescriber interface, prompt template, prompt archive with rollback, meta-prompt controller with three validation layers |
| `framework/executor.py` | Action registry for preprocessing and feature stages, with paradigm/slot metadata |
| `framework/evaluation.py` | Paired differences, exact Wilcoxon, paired-t CI, Holm per family |
| `framework/pipeline.py` | Decision chain: fingerprint → state → graph → memory → gate → prescription → execution |
| `examples/demo.py` | End-to-end example on synthetic signals plus a calibration loop |
| `tests/test_pipeline.py` | 26 tests covering the decision, safety, and execution paths |

## Quick start

```bash
pip install -r requirements.txt
python examples/demo.py
python tests/test_pipeline.py
```

The demo builds a synthetic three-channel record, runs the full chain, and
prints the state, the selected action, its source, and the audit log.

## Build a decision chain

```python
from framework import (
    ActionRegistry, CalibrationMemory, DecisionPipeline, DictKnowledgeGraph,
    GateController, RuleDetector, TemplatePrescriber, band_ratio_fingerprint,
)

registry = ActionRegistry()
registry.register("my_filter", my_filter_fn, paradigm="mi", slot="filtering")
registry.register("my_ica", my_ica_fn, paradigm="mi", slot="artifact")

kg = DictKnowledgeGraph({"line_noise": [("my_filter", 1.0)],
                         "emg": [("my_ica", 1.0), ("my_filter", 0.5)]})

memory = CalibrationMemory(clip_lo=-0.5, clip_hi=1.5,
                           min_subjects=3, strong_gain=0.05, separable_gain=0.02)

detector = RuleDetector({"line_noise": lambda fp: fp[0] > 1.5,
                         "emg": lambda fp: fp[1] > 2.0})

fingerprint = lambda rec, fs: concat_fingerprint(
    my_encoder(rec), band_ratio_fingerprint(rec, fs, 60.0, 2.0, (35.0, 90.0), (6.0, 25.0)))

pipeline = DecisionPipeline(
    fingerprinter=fingerprint, detector=detector, knowledge_graph=kg,
    memory=memory, gates=GateController(strong_gain=0.05), registry=registry,
    context="mi|22|my_dataset", do_no_harm=my_do_no_harm,
    prescriber=None,            # deterministic mode; pass a Prescriber for the LLM arm
)

result = pipeline.decide(record, fs, materials={"note": "..."})
processed = pipeline.execute(record, fs, result.decision)
print(result.decision.action, result.decision.source, result.log)
```

Every threshold is a constructor argument: choose values that fit your
recordings and your decoder.

## Calibrate the memory

```python
from demo import calibrate

written = calibrate(pipeline, records, score_fn=my_decoder_score,
                    subject="s01", memory=memory, fs=fs)
```

`calibrate` runs decide → execute → score → write-back and stores the paired
gain per state and subject. Wrap it in your own fold/repeat loop for the
paper's paired-fold protocol. Feature-stage actions receive `epochs` and
`labels`, and their gain is computed from the feature score against the
no-processing baseline.

## Three configurations

| Configuration | How to build it | Source label |
|---|---|---|
| KG only | empty memory, `prescriber=None` | `kg` |
| KG + memory | filled memory, `prescriber=None` | `memory` / `kg` |
| KG + memory + LLM | filled memory, pass a model as `prescriber` | `llm` |

The LLM receives the memory-rewritten candidate order and chooses inside it;
the post-gate keeps the action in-domain and applies your do-no-harm callback.
Differences between adjacent configurations give the memory and LLM terms.

## Prompt iteration

`MetaPromptController` drives the loop from the supplement:

```python
controller.observe(DecisionRecord("mi|clean", divergence=0.4, median_confidence=0.3))
if controller.should_revise("mi|clean"):
    candidate = controller.propose(base_prompt, evidence, generator)
    passed, checks = controller.validate_all(
        candidate.prompt, executable, parse, results, alpha, action_sets,
        anchors=("mi",), required_fields=("action",),
        valid_rate_before=0.8, valid_rate_after=0.85,
        confidence_before=0.6, confidence_after=0.62)
    if passed:
        controller.adopt(candidate.prompt, "v2", "why", checks)
```

`PromptArchive` keeps each version's text, parent, rationale, validation record,
and timestamp, and `rollback` returns a usable prompt.

## Evaluate

```python
from framework import families_test

report = families_test({
    "primary": {"baseline": paired_gains_baseline, "random": paired_gains_random},
    "secondary": {"mne": paired_gains_mne, "automl": paired_gains_automl},
}, alpha=0.05)
```

Each family is Holm-corrected separately; `wilcoxon_p` uses the exact test by
default and reports a warning when it falls back.

## Extend

- **Methods**: implement `fn(record, fs) -> record` (preprocessing) or
  `fn(epochs, labels) -> score` (feature stage) and register them.
- **Detectors**: add one rule per state in `RuleDetector`.
- **Graph**: fill `DictKnowledgeGraph` with `(action, weight)` edges.
- **Decoder**: write `score_fn(record, subject) -> float` for calibration and
  evaluation.
- **Model**: implement `Prescriber` (or wrap your model in `TemplatePrescriber`).

`REFERENCES.md` lists method-family papers you can draw on.

## License and citation

MIT-licensed (see `LICENSE`). Citation metadata is in `CITATION.cff`; GitHub
renders a "Cite this repository" button from it. The paper cites this
repository directly, so `main` is kept as the version of record: cite the
repository URL, not a tag.
