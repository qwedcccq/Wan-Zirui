"""Decision chain assembly.

Order of operations (paper §2.4):

1. compress the record into a fingerprint;
2. detect the state set S(x) and form its single-state keys;
3. retrieve candidates and a mechanistic order from the knowledge graph;
4. let calibration memory rewrite the order and return a recommendation a_B
   with its evidence strength G; a memory answer below ``min_subjects`` falls
   back to the skeleton;
5. pre-gate: with strong evidence and reuse enabled, execute a_B directly;
   otherwise consult the prescriber — or, in deterministic mode
   (``prescriber=None``), execute the memory-rewritten backbone a_B;
6. post-gate: strong evidence keeps a_B; otherwise adopt the prescription when
   it is executable and passes do-no-harm, else the knowledge-graph action,
   else no processing;
7. execute through the registry.  Selection and execution stay separate.

Safety rules enforced here:

- memory never expands the candidate set; it only reorders KG candidates;
- gain and sample size come from the same contexts (exact first, parallel
  second) — see ``CalibrationMemory.answer``;
- low-confidence memory does not take effect: the backbone is the skeleton;
- indistinguishable gains fall back to the default action;
- an empty knowledge graph yields no processing, which executes as keep-as-is;
- the do-no-harm callback is required; when it is missing or returns False,
  an LLM action is not adopted and the chain falls back to the KG action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .executor import ActionRegistry
from .fingerprint import Fingerprinter
from .gates import Decision, GateController, Source
from .knowledge_graph import KnowledgeGraph
from .memory import CalibrationMemory
from .prescription import DecisionRecord, MetaPromptController, Prescriber
from .state_set import StateDetector


@dataclass
class PipelineDecision:
    decision: Decision
    state_key: str
    candidates: List[str] = field(default_factory=list)
    memory_gain: Optional[float] = None
    log: Dict[str, object] = field(default_factory=dict)


class DecisionPipeline:
    def __init__(
        self,
        fingerprinter: Fingerprinter,
        detector: StateDetector,
        knowledge_graph: KnowledgeGraph,
        memory: CalibrationMemory,
        gates: GateController,
        registry: ActionRegistry,
        context: str,
        do_no_harm: Optional[Callable[[str], bool]] = None,
        prescriber: Optional[Prescriber] = None,
        reuse: bool = True,
        default_action: Optional[str] = None,
        meta_controller: Optional[MetaPromptController] = None,
    ):
        if do_no_harm is None:
            raise ValueError(
                "do_no_harm callback is required; return False for an action "
                "that must not be adopted (fail closed)"
            )
        self.fingerprinter = fingerprinter
        self.detector = detector
        self.kg = knowledge_graph
        self.memory = memory
        self.prescriber = prescriber
        self.gates = gates
        self.registry = registry
        self.context = context
        self.do_no_harm = do_no_harm
        self.reuse = reuse
        self.default_action = default_action
        self.meta_controller = meta_controller

    # ------------------------------------------------------------------
    def _kg_order(self, state_keys: List[str], candidates: List[str]) -> List[str]:
        if hasattr(self.kg, "candidates_multi"):
            ordered = [c for c in self.kg.candidates_multi(state_keys) if c in candidates]
            if ordered:
                return ordered
        merged: List[str] = []
        for key in state_keys:
            for action in self.kg.rank(key, candidates):
                if action not in merged:
                    merged.append(action)
        return merged or list(candidates)

    def _memory_backbone(
        self, state_keys: List[str], candidates: List[str]
    ) -> Tuple[Optional[str], Optional[float], int, List[str]]:
        """Memory-rewritten backbone (a_B), or None when low confidence.

        Returns ``(action, gain, n_subjects, used_contexts)``.  When the best
        answer has fewer than ``min_subjects`` subjects, the answer is
        discarded and the caller uses the skeleton.
        """
        answers = []
        for action in candidates:
            answer = self.memory.answer_any(self.context, state_keys, action)
            if answer is not None:
                answers.append((action, answer))
        if not answers:
            return None, None, 0, []
        confident = [(a, ans) for a, ans in answers
                     if ans.n_subjects >= self.memory.min_subjects]
        if not confident:
            best = max(answers, key=lambda t: t[1].gain)
            return None, None, best[1].n_subjects, list(best[1].contexts)
        confident.sort(key=lambda t: t[1].gain, reverse=True)
        action, answer = confident[0]

        anchor = (self.default_action
                  if self.default_action in candidates else candidates[0])
        anchor_answer = self.memory.answer_any(self.context, state_keys, anchor)
        if (anchor_answer is not None and action != anchor
                and anchor_answer.n_subjects >= self.memory.min_subjects
                and answer.gain - anchor_answer.gain < self.memory.separable_gain):
            action, answer = anchor, anchor_answer
        return action, answer.gain, answer.n_subjects, list(answer.contexts)

    def _memory_order(
        self, state_keys: List[str], kg_order: List[str],
        backbone: Optional[str],
    ) -> List[str]:
        """Memory-rewritten candidate order: backbone first, then by memory gain.

        Only answers that reach ``min_subjects`` may reorder the list; a
        low-confidence memory entry never moves ahead of the skeleton.
        """
        if backbone is None:
            return list(kg_order)
        scored = []
        for action in kg_order:
            answer = self.memory.answer_any(self.context, state_keys, action)
            if answer is not None and answer.n_subjects >= self.memory.min_subjects:
                scored.append((action, answer.gain))
        scored.sort(key=lambda t: t[1], reverse=True)
        order = [a for a, _ in scored]
        order += [a for a in kg_order if a not in order]
        order = [backbone] + [a for a in order if a != backbone]
        return order

    # ------------------------------------------------------------------
    def decide(self, record, fs: float, materials: Optional[Dict[str, str]] = None,
               divergence: Optional[float] = None,
               median_confidence: Optional[float] = None,
               condition_key: Optional[str] = None) -> PipelineDecision:
        signal = None
        if divergence is not None and median_confidence is not None:
            signal = DecisionRecord(condition_key or self.context,
                                    divergence, median_confidence)
            if self.meta_controller is not None:
                self.meta_controller.observe(signal)

        fingerprint = self.fingerprinter(record, fs)
        state = self.detector(fingerprint)
        state_key = state.key()
        state_keys = state.keys()
        executable = self.registry.executable()

        # 3. knowledge graph: candidates and mechanistic order (no fallback)
        if hasattr(self.kg, "candidates_multi"):
            raw_candidates = self.kg.candidates_multi(state_keys)
        else:
            raw_candidates = []
            for key in state_keys:
                raw_candidates.extend(self.kg.candidates(key))
        kg_candidates = [c for c in dict.fromkeys(raw_candidates)
                         if c in self.registry.candidates()]
        executable_candidates = [c for c in kg_candidates if c in executable]
        if not kg_candidates or not executable_candidates:
            decision = Decision("none", Source.NONE, False, False)
            return PipelineDecision(
                decision, state_key, [], None,
                {"path": "no_candidates", "strong": False, "reused": False,
                 "backbone_action": None, "memory_action": None,
                 "kg_action": None, "llm_action": None,
                 "memory_provenance": [], "divergence": None,
                 "median_confidence": None},
            )
        kg_order = self._kg_order(state_keys, kg_candidates)
        kg_executable_order = [c for c in kg_order if c in executable]
        kg_top = kg_executable_order[0]

        # 4. memory rewrites the order; low confidence falls back to skeleton
        memory_action, memory_gain, n_subjects, memory_contexts = self._memory_backbone(
            state_keys, executable_candidates
        )
        memory_order = self._memory_order(state_keys, kg_executable_order, memory_action)

        if memory_action is not None:
            a_b, gain_b, b_source = memory_action, memory_gain, Source.MEMORY
        else:
            a_b, gain_b, b_source = kg_top, None, Source.KG
            n_subjects = 0

        strong = self.gates.is_strong(
            gain_b, a_b, executable, b_source,
            n_subjects=n_subjects, min_subjects=self.memory.min_subjects,
        )

        provenance = ([self.memory.provenance(self.context, state_keys, a_b)]
                      if memory_action is not None else [])

        # 5. pre-gate
        if self.gates.pre_gate(strong, self.reuse):
            decision = Decision(a_b, Source.MEMORY, True, True)
            return PipelineDecision(
                decision, state_key, memory_order, gain_b,
                {"path": "memory_reuse", "strong": True, "reused": True,
                 "backbone_action": a_b, "memory_action": memory_action,
                 "kg_action": kg_top, "llm_action": None,
                 "memory_contexts": memory_contexts,
                 "memory_provenance": provenance,
                 "divergence": signal.divergence if signal else None,
                 "median_confidence": signal.median_confidence if signal else None},
            )

        # 6. prescription (or deterministic replacement) and post-gate
        llm_action = None
        if self.prescriber is None:
            rule_source = Source.MEMORY if memory_action is not None else Source.KG
            decision = self.gates.post_gate(
                strong, a_b, a_b, kg_top, executable,
                do_no_harm=self.do_no_harm, reuse=self.reuse,
                rule_source=rule_source, candidates=memory_order,
            )
        else:
            llm_action = self.prescriber(state_key, memory_order, materials or {})
            decision = self.gates.post_gate(
                strong, a_b, llm_action, kg_top, executable,
                do_no_harm=self.do_no_harm, reuse=self.reuse, candidates=memory_order,
            )
        return PipelineDecision(
            decision, state_key, memory_order, gain_b,
            {"path": "prescription", "strong": strong, "reused": decision.reused,
             "backbone_action": a_b, "memory_action": memory_action,
             "kg_action": kg_top, "llm_action": llm_action,
             "llm_adopted": decision.source == Source.LLM,
             "memory_contexts": memory_contexts,
             "memory_provenance": provenance,
             "divergence": signal.divergence if signal else None,
             "median_confidence": signal.median_confidence if signal else None},
        )

    # ------------------------------------------------------------------
    def execute(self, record, fs: float, decision: Decision,
                epochs=None, labels=None):
        """Run the selected action according to its stage.

        - no processing: keep-as-is (``record.copy()``);
        - preprocessing: ``registry.execute``;
        - feature: ``registry.execute_feature`` (``epochs``/``labels`` required).
        """
        if decision.action == "none" or decision.source == Source.NONE:
            return record.copy()
        spec = self.registry.get(decision.action)
        if spec.fn is None:
            raise RuntimeError(f"action '{decision.action}' is decision-only (no tool)")
        if spec.stage == "feature":
            if epochs is None or labels is None:
                raise ValueError("feature-stage execution requires epochs and labels")
            return self.registry.execute_feature(decision.action, epochs, labels)
        return self.registry.execute(decision.action, record, fs)
