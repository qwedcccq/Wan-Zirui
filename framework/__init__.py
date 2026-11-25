"""Framework package: state-conditioned EEG preprocessing selection.

The package provides interfaces and control flow only.  Methods, detectors,
graph contents, and measured gains are supplied by the reader.
"""

from .evaluation import family_test, families_test, holm, paired_differences, paired_t_ci, wilcoxon_p
from .executor import ActionFn, ActionRegistry, ActionSpec, FeatureActionFn
from .fingerprint import Fingerprinter, band_ratio_fingerprint, concat_fingerprint
from .gates import Decision, GateController, Source
from .knowledge_graph import DictKnowledgeGraph, KnowledgeGraph
from .memory import CalibrationMemory, Entry, MemoryAnswer
from .pipeline import DecisionPipeline, PipelineDecision
from .prescription import (
    DecisionRecord,
    GenerationResult,
    MetaPromptController,
    Prescriber,
    PromptArchive,
    PromptVersion,
    TemplatePrescriber,
)
from .state_set import (
    DETERMINISTIC_CANDIDATES,
    ERP_DETERMINISTIC_CANDIDATES,
    ERP_HINT_CANDIDATES,
    HINT_CANDIDATES,
    RuleDetector,
    StateDetector,
    StateSet,
    STATE_LABELS,
    STATES,
)

__all__ = [
    "ActionFn",
    "ActionRegistry",
    "ActionSpec",
    "FeatureActionFn",
    "CalibrationMemory",
    "DETERMINISTIC_CANDIDATES",
    "ERP_DETERMINISTIC_CANDIDATES",
    "ERP_HINT_CANDIDATES",
    "Decision",
    "DecisionPipeline",
    "DecisionRecord",
    "DictKnowledgeGraph",
    "Entry",
    "GateController",
    "GenerationResult",
    "HINT_CANDIDATES",
    "KnowledgeGraph",
    "MemoryAnswer",
    "MetaPromptController",
    "PipelineDecision",
    "Prescriber",
    "PromptArchive",
    "PromptVersion",
    "RuleDetector",
    "Source",
    "StateDetector",
    "StateSet",
    "STATE_LABELS",
    "STATES",
    "TemplatePrescriber",
    "band_ratio_fingerprint",
    "concat_fingerprint",
    "family_test",
    "families_test",
    "holm",
    "paired_differences",
    "paired_t_ci",
    "wilcoxon_p",
]
