"""Pre-gate and post-gate.

    Strong(s, a_B) = m(s) != empty
                     and a_B in A
                     and src(a_B) == memory
                     and G(s, a_B) >= strong_gain
                     and n(s, a_B) >= min_subjects

    a  = a_B            if Strong and reuse
         Consult(s)     otherwise

    a* = a_B                   if Strong
         a_L                   if not Strong and a_L in A and do-no-harm passes
         a_KG                  otherwise

A deterministic replacement (no LLM) returns a_B directly and is labelled with
the requesting source (``rule_source``), so a rule-based configuration is not
misreported as an LLM decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class Source(str, Enum):
    MEMORY = "memory"
    LLM = "llm"
    KG = "kg"
    RULE = "rule"
    NONE = "none"


@dataclass(frozen=True)
class Decision:
    action: str
    source: Source
    strong: bool
    reused: bool


class GateController:
    """Pre-gate and post-gate for one decision."""

    def __init__(self, strong_gain: float):
        self.strong_gain = float(strong_gain)

    # ------------------------------------------------------------- pre-gate
    def is_strong(
        self,
        memory_gain: Optional[float],
        action: str,
        executable: set,
        action_source: Source,
        n_subjects: int,
        min_subjects: int,
    ) -> bool:
        """Strong evidence; sample size is required and checked first."""
        if n_subjects < min_subjects:
            return False
        return (
            memory_gain is not None
            and action in executable
            and action_source == Source.MEMORY
            and memory_gain >= self.strong_gain
        )

    def pre_gate(self, strong: bool, reuse: bool) -> bool:
        """True when the memory recommendation is executed without consulting."""
        return bool(strong and reuse)

    # ------------------------------------------------------------ post-gate
    def post_gate(
        self,
        strong: bool,
        memory_action: str,
        llm_action: Optional[str],
        kg_action: Optional[str],
        executable: set,
        do_no_harm: Optional[Callable[[str], bool]],
        reuse: bool,
        rule_source: Optional[Source] = None,
    ) -> Decision:
        """Resolve the final action.

        - strong evidence: the memory action is final; an LLM deviation is
          rejected as a safety fallback.  ``reused`` records whether the
          pre-gate actually skipped consultation.
        - ``rule_source`` given (deterministic mode): return ``memory_action``
          (the backbone) with that source, without consulting a model.
        - otherwise: adopt the LLM action when it is executable and passes
          do-no-harm; else the knowledge-graph action; else no processing.
        """
        if strong:
            return Decision(memory_action, Source.MEMORY, True, reuse)

        if rule_source is not None:
            return Decision(memory_action, rule_source, False, False)

        if llm_action is not None and llm_action in executable:
            # fail closed: without a do-no-harm callback the LLM action is not adopted
            if do_no_harm is not None and do_no_harm(llm_action):
                return Decision(llm_action, Source.LLM, False, False)
        if kg_action is not None and kg_action in executable:
            return Decision(kg_action, Source.KG, False, False)
        return Decision("none", Source.NONE, False, False)
