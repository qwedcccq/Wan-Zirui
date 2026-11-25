# Reproducible framework: state-conditioned EEG preprocessing selection

This package contains the **framework only** described in the paper. It contains:

- no experimental data,
- no prior-knowledge tables (no knowledge-graph contents, no method registry, no measured
  gains),
- no concrete preprocessing methods.

Every method, state detector, knowledge-graph edge, and action implementation is left for
the reader to supply. `REFERENCES.md` is an optional reading list of method-family papers;
it is not a method registry and carries no parameter from the paper.

## What the framework provides

| Module | Provides | Reader supplies |
|---|---|---|
| `framework/fingerprint.py` | Interface and generic band-ratio example | Encoder, thresholds |
| `framework/state_set.py` | The eight state names and set semantics | Detectors for each state |
| `framework/knowledge_graph.py` | Interface and empty implementation | Nodes, edges, ranking |
| `framework/memory.py` | Long-format table, lookup, admission rule | Measured gains |
| `framework/gates.py` | Pre-gate and post-gate logic | Threshold values |
| `framework/prescription.py` | Prescriber interface and a generic prompt template | Model, prompt engineering |
| `framework/executor.py` | Action registry and execution interface | Method implementations |
| `framework/evaluation.py` | Paired-fold scoring, Wilcoxon, Holm | Datasets and decoders |
| `framework/pipeline.py` | Decision chain assembly | Everything above |

## Quick start

```bash
python examples/demo.py
```

The demo uses synthetic signals and two no-op placeholders (`rec.copy()`) so the chain runs
end to end without any experimental data. It is not a reproduction of the paper's results;
it only shows how the pieces connect.

## How to rebuild a comparable setting

The framework gives the control flow; the numbers come from your methods.  A minimal
path to a comparable experiment:

1. **Fingerprint** — wrap your frozen encoder and call `concat_fingerprint(encoder(x),
   band_ratio_fingerprint(x, fs, ...))`.  Eight dimensions on discriminative paradigms,
   ten on the ERP path.
2. **Detector** — write one rule per state in `RuleDetector`; the thresholds are yours.
3. **Knowledge graph** — fill `DictKnowledgeGraph` with `(action, weight)` edges for
   the states you detect.
4. **Actions** — implement your methods and register them in `ActionRegistry`; the
   registry is the executable set.
5. **Judge** — write `score_fn(record, subject) -> float` that decodes the processed
   record on held-out folds.
6. **Calibration loop** — `examples.demo.calibrate` shows the order: decide, execute,
   score against no processing, write the paired gain back per state and subject.
7. **Three configurations** — the paper's attribution uses exactly these:
   - *KG only*: empty memory, `prescriber=None` (deterministic mode). The final action
     is the knowledge-graph top-1 and the source is `kg`.
   - *KG + memory*: memory filled by step 6, `prescriber=None`. The final action is the
     memory-rewritten backbone and the source is `memory` (or `kg` for low-confidence
     states).
   - *KG + memory + LLM*: pass your model as `prescriber`; the LLM receives the
     memory-rewritten candidate order and may deviate within it.
   Differences between adjacent configurations are the memory and LLM terms.  Source
   labels stay distinct (`kg`, `memory`, `llm`), so the three terms can be counted.
8. **Evaluation** — collect paired fold differences and call `family_test` once per
   pre-specified family (for example primary and secondary controls).

The demo runs this chain with synthetic signals and no-op actions.  Its bands,
thresholds, and outputs are illustrative and are not the paper's values.

## Prompt iteration

`MetaPromptController` implements the loop described in the supplement:

1. `observe(DecisionRecord(condition_key, divergence, median_confidence))` after each
   decision; an anomaly is a divergence above the threshold or a median confidence below it.
2. `should_revise(condition_key)` fires after the configured number of consecutive anomalies.
3. `propose(base_prompt, evidence, generator)` — the generator runs the four steps
   (normalise evidence, diagnose, design item by item, validate the parse structure) and
   returns `GenerationResult(diagnosis, changes, prompt)`.
4. `validate_all(...)` checks the constraint, downstream, and self-evaluation layers;
   `adopt` refuses a candidate whose `all_passed` is False.
5. `adopt(candidate_prompt, ...)` archives the prompt with parent, rationale, validation,
   and timestamp; `rollback(version_id)` returns a usable prompt.

`DecisionPipeline.decide` forwards `divergence`, `median_confidence`, and `condition_key`
to the controller when they are supplied, so the loop is driven from the same call.

`examples.demo.calibrate` is a single-pass scaffold: it scores one record at a time.
Wrap it in your own fold loop to obtain the paired-fold protocol of the paper.

## What is deliberately absent

- The paper's action list, knowledge-graph contents, and measured-gain tables.
- Any dataset path or subject identifier.
- Any numerical threshold from the paper. All thresholds are constructor arguments with
  no default; you choose them.

Literature-recommended values (for example 50/60 Hz line frequency, 40-100 Hz EMG
power, 8-30 Hz mu+beta) appear only as comments next to parameters and are marked
with their source. They are not defaults and do not reproduce the paper.
