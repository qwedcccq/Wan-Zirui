"""Prescriber interface and meta-prompt iteration.

Two things live here:

1. ``Prescriber`` — the interface a decision model must satisfy.  No prompt
   text from the paper is included; ``PROMPT_TEMPLATE`` is a structural
   template the reader fills.
2. ``MetaPromptController`` — the prompt-iteration loop: anomaly detection,
   candidate generation, three validation layers, version archive, rollback.

The loop is the paper's mechanism (S23 / Table S18) without its values.  All
thresholds are constructor arguments without defaults.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Prescriber
# --------------------------------------------------------------------------
class Prescriber:
    """Decision model interface: ``state + candidates + materials -> action``."""

    def __call__(
        self,
        state_key: str,
        candidates: Sequence[str],
        materials: Dict[str, str],
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError


PROMPT_TEMPLATE = """Role: {role}
Materials:
{materials}
Candidate actions (choose exactly one identifier):
{candidates}
Output: {schema}
"""


class TemplatePrescriber(Prescriber):
    """Build a prompt from the template and call a user-supplied model.

    ``model`` is any callable ``prompt -> action_identifier``.  The framework
    does not ship a model or a prompt body.
    """

    def __init__(
        self,
        model: Callable[[str], str],
        role: str,
        schema: str,
    ):
        self.model = model
        self.role = role
        self.schema = schema

    def __call__(self, state_key, candidates, materials) -> str:
        prompt = PROMPT_TEMPLATE.format(
            role=self.role,
            materials="\n".join(f"- {k}: {v}" for k, v in materials.items()),
            candidates="\n".join(f"- {c}" for c in candidates),
            schema=self.schema,
        )
        return self.model(prompt)


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------
@dataclass
class PromptVersion:
    version_id: str
    prompt: str
    parent: Optional[str]
    source_label: str
    rationale: str
    validation: Dict[str, object]
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))


class PromptArchive:
    """Versioned prompt store with rollback.

    Each version keeps the prompt text, so a rollback returns a usable prompt.
    """

    def __init__(self) -> None:
        self._versions: Dict[str, PromptVersion] = {}
        self.current: Optional[str] = None
        self._seq = 0

    def add(
        self,
        source_label: str,
        prompt: str,
        parent: Optional[str],
        rationale: str,
        validation: Dict[str, object],
    ) -> PromptVersion:
        self._seq += 1
        version = PromptVersion(
            version_id=f"v{self._seq}",
            prompt=prompt,
            parent=parent,
            source_label=source_label,
            rationale=rationale,
            validation=dict(validation),
        )
        self._versions[version.version_id] = version
        self.current = version.version_id
        return version

    def get(self, version_id: Optional[str] = None) -> Optional[PromptVersion]:
        return self._versions.get(version_id or self.current)

    def rollback(self, version_id: str) -> PromptVersion:
        if version_id not in self._versions:
            raise KeyError(version_id)
        self.current = version_id
        return self._versions[version_id]

    def history(self) -> List[PromptVersion]:
        return [self._versions[k] for k in sorted(self._versions, key=lambda s: int(s[1:]))]


# --------------------------------------------------------------------------
# Meta-prompt iteration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionRecord:
    """One decision's confidence signals."""

    condition_key: str      # e.g. "<paradigm>|<signal-quality grade>"
    divergence: float       # |model confidence - deterministic score|
    median_confidence: float


@dataclass
class GenerationResult:
    """Output of one generation step (four-step procedure)."""

    diagnosis: str          # cause identified from the evidence
    changes: List[str]      # improvements designed item by item
    prompt: str             # candidate prompt text


class MetaPromptController:
    """Anomaly-driven prompt iteration with three validation layers.

    Parameters (all required, no defaults)
    --------------------------------------
    trigger_count : int
        Consecutive anomalous decisions required before a revision is tried.
    divergence_threshold : float
        A decision is anomalous when ``divergence`` exceeds this.
    confidence_threshold : float
        A decision is anomalous when ``median_confidence`` falls below this.
    generation_temperature : float
        Temperature passed to the generator callable.
    min_improved_conditions : int
        Downstream layer: at least this many conditions must improve.
    max_worsened_conditions : int
        Downstream layer: at most this many conditions may worsen.
    jaccard_threshold : float
        Self-evaluation layer: action-set similarity must reach this.
    """

    def __init__(
        self,
        archive: PromptArchive,
        trigger_count: int,
        divergence_threshold: float,
        confidence_threshold: float,
        generation_temperature: float,
        min_improved_conditions: int,
        max_worsened_conditions: int,
        jaccard_threshold: float,
    ):
        self.archive = archive
        self.trigger_count = int(trigger_count)
        self.divergence_threshold = float(divergence_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.generation_temperature = float(generation_temperature)
        self.min_improved_conditions = int(min_improved_conditions)
        self.max_worsened_conditions = int(max_worsened_conditions)
        self.jaccard_threshold = float(jaccard_threshold)
        self._anomalies: Dict[str, int] = {}

    # ---------------------------------------------------------- anomaly
    def observe(self, record: DecisionRecord) -> bool:
        """Record one decision; return True when it counts as anomalous."""
        anomalous = (
            record.divergence > self.divergence_threshold
            or record.median_confidence < self.confidence_threshold
        )
        key = record.condition_key
        self._anomalies[key] = self._anomalies.get(key, 0) + 1 if anomalous else 0
        return anomalous

    def should_revise(self, condition_key: str) -> bool:
        return self._anomalies.get(condition_key, 0) >= self.trigger_count

    # --------------------------------------------------------- generation
    def propose(
        self,
        base_prompt: str,
        evidence: str,
        generator: Callable[[str, float], object],
    ) -> "GenerationResult":
        """Ask the generator for a candidate prompt.

        ``generator`` receives ``(base_prompt + evidence, temperature)`` and
        returns either a :class:`GenerationResult` or a prompt string.  The
        four-step procedure (normalise evidence, diagnose the cause, design
        improvements item by item, validate the parse structure) belongs to
        the generator; returning a :class:`GenerationResult` makes the
        diagnosis and the itemised changes auditable.
        """
        raw = generator(f"{base_prompt}\n\n{evidence}", self.generation_temperature)
        if isinstance(raw, GenerationResult):
            return raw
        return GenerationResult(diagnosis="", changes=[], prompt=str(raw))

    # ---------------------------------------------------------- validation
    def validate_structural(
        self, candidate: str, executable: set, parse: Callable[[str], Optional[str]]
    ) -> bool:
        """Layer 1: the prompt must yield a parseable identifier in the set."""
        try:
            action = parse(candidate)
        except Exception:
            return False
        return action is not None and action in executable

    def validate_constraint(
        self,
        candidate: str,
        anchors: Sequence[str] = (),
        required_fields: Sequence[str] = (),
    ) -> bool:
        """Layer 1 (constraint part): scenario anchors and output-contract fields.

        ``anchors`` are keywords that identify the scenario (for example the
        state or the paradigm); ``required_fields`` are the output fields the
        prompt must ask for.  Both checks are case-insensitive substring
        checks, so the reader owns the exact wording.
        """
        text = candidate.lower()
        return (all(a.lower() in text for a in anchors)
                and all(f.lower() in text for f in required_fields))

    def validate_downstream(
        self, results: Sequence[Tuple[str, float, float]], alpha: float
    ) -> bool:
        """Layer 2: conditions of ``(name, delta, p)``.

        A condition improves when ``delta > 0`` and ``p < alpha``; the same
        rule with ``delta < 0`` counts as worsening.  ``p`` values are
        expected to be corrected already.
        """
        improved = sum(1 for _, d, p in results if d > 0 and p < alpha)
        worsened = sum(1 for _, d, p in results if d < 0 and p < alpha)
        return (
            improved >= self.min_improved_conditions
            and worsened <= self.max_worsened_conditions
        )

    def validate_self_eval(
        self,
        action_sets: Sequence[Sequence[str]],
        valid_rate_before: Optional[float] = None,
        valid_rate_after: Optional[float] = None,
        confidence_before: Optional[float] = None,
        confidence_after: Optional[float] = None,
        tolerance: float = 0.0,
    ) -> bool:
        """Layer 3: action-set similarity, success rate, and confidence.

        - mean pairwise Jaccard similarity of the action sets (at least two);
        - the revised version is not lower than the pre-revision version in
          valid-decision success rate and median module confidence, within
          ``tolerance``.
        """
        sets = [set(s) for s in action_sets]
        if len(sets) < 2:
            return False
        sims: List[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                sims.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        jaccard_ok = (sum(sims) / len(sims)) >= self.jaccard_threshold
        rate_ok = (valid_rate_before is None or valid_rate_after is None
                   or valid_rate_after >= valid_rate_before - tolerance)
        conf_ok = (confidence_before is None or confidence_after is None
                   or confidence_after >= confidence_before - tolerance)
        return jaccard_ok and rate_ok and conf_ok

    def validate_all(
        self,
        candidate_prompt: str,
        executable: set,
        parse: Callable[[str], Optional[str]],
        results: Sequence[Tuple[str, float, float]],
        alpha: float,
        action_sets: Sequence[Sequence[str]],
        anchors: Sequence[str] = (),
        required_fields: Sequence[str] = (),
    ) -> Tuple[bool, Dict[str, bool]]:
        """Run the three validation layers and return ``(passed, detail)``."""
        checks = {
            "structural": self.validate_structural(candidate_prompt, executable, parse),
            "constraint": self.validate_constraint(candidate_prompt, anchors, required_fields),
            "downstream": self.validate_downstream(results, alpha),
            "self_eval": self.validate_self_eval(action_sets),
        }
        return all(checks.values()), checks

    # ------------------------------------------------------------- adopt
    def adopt(
        self,
        candidate_prompt: str,
        source_label: str,
        rationale: str,
        validation: Dict[str, object],
    ) -> PromptVersion:
        """Archive a candidate; refuse when validation says it did not pass."""
        if validation.get("all_passed") is False:
            raise ValueError("candidate did not pass all three validation layers")
        parent = self.archive.current
        return self.archive.add(source_label, candidate_prompt, parent, rationale, validation)

    def rollback(self, version_id: str) -> PromptVersion:
        return self.archive.rollback(version_id)
