"""Tests for the framework, including the safety and execution paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from demo import build_demo, synthetic_record  # noqa: E402
from framework import (  # noqa: E402
    ActionRegistry,
    CalibrationMemory,
    DecisionPipeline,
    DictKnowledgeGraph,
    GateController,
    GenerationResult,
    PromptArchive,
    TemplatePrescriber,
    holm,
    paired_t_ci,
    wilcoxon_p,
)

CONTEXT = "demo|3|synthetic"
_USE_DEFAULT = object()


def _parts():
    return build_demo()


def _pipeline(memory, kg=None, prescriber=_USE_DEFAULT, do_no_harm=None,
              context=CONTEXT, registry=None):
    fs, registry0, fingerprinter, detector, kg0, _, prescriber0, gates = _parts()
    if prescriber is _USE_DEFAULT:
        prescriber = prescriber0
    return fs, DecisionPipeline(
        fingerprinter=fingerprinter,
        detector=detector,
        knowledge_graph=kg if kg is not None else kg0,
        memory=memory,
        prescriber=prescriber,
        gates=gates,
        registry=registry if registry is not None else registry0,
        context=context,
        do_no_harm=do_no_harm or (lambda action: True),
        default_action="placeholder_a",
    )


def _memory(min_subjects=2, strong_gain=0.10, separable=0.05):
    return CalibrationMemory(clip_lo=-0.5, clip_hi=1.5,
                             min_subjects=min_subjects, strong_gain=strong_gain,
                             separable_gain=separable)


# ---------------------------------------------------------------- basics
def test_empty_memory_consultation():
    fs, pipeline = _pipeline(_memory())
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.log["memory_action"] is None
    assert out.decision.action in pipeline.registry.candidates()


def test_strong_memory_reuse():
    fs, _, _, _, _, _, _, _ = _parts()
    memory = _memory()
    for subject in ("s1", "s2"):
        memory.write_back(CONTEXT, "line_noise", "placeholder_b", subject, 0.20)
    fs, pipeline = _pipeline(memory)
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.strong is True
    assert out.decision.reused is True
    assert out.decision.action == "placeholder_b"


def test_low_confidence_not_strong_and_backbone_is_skeleton():
    fs, _, _, _, _, _, _, _ = _parts()
    memory = _memory(min_subjects=3)
    memory.write_back(CONTEXT, "line_noise", "placeholder_b", "s1", 0.50)
    fs, pipeline = _pipeline(memory)
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.strong is False
    assert out.log["backbone_action"] == out.log["kg_action"]


def test_parallel_does_not_inflate_exact_sample():
    fs, _, _, _, _, _, _, _ = _parts()
    memory = _memory(min_subjects=3)
    memory.write_back(CONTEXT, "line_noise", "placeholder_b", "s1", 0.50)
    memory.write_back("demo|3|other", "line_noise", "placeholder_b", "s2", 0.50)
    memory.write_back("demo|3|other", "line_noise", "placeholder_b", "s3", 0.50)
    fs, pipeline = _pipeline(memory)
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.strong is False


def test_do_no_harm_blocks_llm():
    fs, registry, fingerprinter, detector, _, _, _, gates = _parts()
    kg = DictKnowledgeGraph({
        "line_noise": [("placeholder_a", 1.0), ("placeholder_b", 0.5)],
        "normal_unknown": [("placeholder_a", 1.0)],
    })
    prescriber = TemplatePrescriber(lambda prompt: "placeholder_b", "demo", "{}")
    fs, pipeline = _pipeline(_memory(), kg=kg, prescriber=prescriber,
                             do_no_harm=lambda action: action != "placeholder_b")
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.source.value == "kg"
    assert out.decision.action == "placeholder_a"


def test_inseparable_backbone_falls_back():
    fs, _, _, _, _, _, _, _ = _parts()
    memory = _memory(separable=0.10)
    for subject in ("s1", "s2"):
        memory.write_back(CONTEXT, "line_noise", "placeholder_b", subject, 0.06)
        memory.write_back(CONTEXT, "line_noise", "placeholder_a", subject, 0.02)
    fs, pipeline = _pipeline(memory)
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.log["backbone_action"] == "placeholder_a"


def test_parallel_context_fallback_and_provenance():
    fs, _, _, _, _, _, _, _ = _parts()
    memory = _memory()
    for subject in ("s1", "s2"):
        memory.write_back("demo|3|other", "line_noise", "placeholder_b", subject, 0.20)
    fs, pipeline = _pipeline(memory)
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.strong is True
    prov = out.log["memory_provenance"][0]
    assert prov["n_subjects"] == 2
    assert prov["contexts"] == ["demo|3|other"]


# ------------------------------------------------------------ execution
def test_no_processing_executes_as_identity():
    fs, pipeline = _pipeline(_memory(), kg=DictKnowledgeGraph({}))
    record = synthetic_record(fs)
    out = pipeline.decide(record, fs)
    assert out.decision.action == "none"
    executed = pipeline.execute(record, fs, out.decision)
    assert np.array_equal(executed, record)


def test_feature_stage_execution():
    fs, _, fingerprinter, detector, _, _, _, gates = _parts()
    registry = ActionRegistry()
    registry.register("feature_a", lambda epochs, labels: 0.5, stage="feature")
    kg = DictKnowledgeGraph({"line_noise": [("feature_a", 1.0)]})
    pipeline = DecisionPipeline(
        fingerprinter=fingerprinter, detector=detector, knowledge_graph=kg,
        memory=_memory(), gates=gates, registry=registry, context=CONTEXT,
        do_no_harm=lambda action: True, prescriber=None,
    )
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.action == "feature_a"
    # deterministic mode: the backbone is reported with its own source (KG here)
    assert out.decision.source.value == "kg"
    score = pipeline.execute(None, fs, out.decision,
                             epochs=np.zeros((2, 3, 10)), labels=np.array([0, 1]))
    assert score == 0.5


def test_rule_mode_is_not_labelled_llm():
    fs, _, _, _, _, _, _, _ = _parts()
    fs, pipeline = _pipeline(_memory(), prescriber=None)
    out = pipeline.decide(synthetic_record(fs), fs)
    assert out.decision.source.value in {"kg", "memory", "rule"}


# -------------------------------------------------------- package/API
def test_generation_result_importable():
    assert GenerationResult is not None


def test_prompt_archive_rollback_keeps_prompt():
    archive = PromptArchive()
    v1 = archive.add("draft", "prompt one", None, "initial", {})
    v2 = archive.add("revision", "prompt two", v1.version_id, "improved", {})
    rolled = archive.rollback(v1.version_id)
    assert rolled.prompt == "prompt one"
    assert v2.prompt == "prompt two"


def test_by_paradigm_excludes_unlabelled():
    registry = ActionRegistry()
    registry.register("unlabelled", lambda rec, fs: rec)
    registry.register("mi_action", lambda rec, fs: rec, paradigm="mi")
    assert registry.by(paradigm="mi") == ["mi_action"]
    assert registry.by(paradigm="ssvep") == []


def test_holm_known_values():
    adjusted = holm([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_paired_ci_and_wilcoxon():
    lo, hi = paired_t_ci([1.0, 2.0, 3.0, 4.0], 0.05)
    assert lo < hi
    assert wilcoxon_p([0.0, 0.0, 0.0]) == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
