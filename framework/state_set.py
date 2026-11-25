"""State set definitions.

The framework treats a recording as belonging to a *set* of data states rather
than to a single class.  This module defines the vocabulary and the set
semantics only; no detector, threshold, or prior probability is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

#: The eight state names used by the paper.  They are vocabulary, not prior
#: knowledge: the reader decides how each state is detected.
#: ``channel_corruption`` is the main-text name; figures label it "bad channel".
STATES = (
    "normal_unknown",
    "line_noise",
    "channel_corruption",
    "emg",
    "low_snr",
    "drift",
    "spike",
    "eog",
)

#: Three-channel detector: four deterministic states and four hint states.
#: The ten-dimensional ERP detector returns six deterministic states (adding
#: drift and a blink criterion) and two hints; the reader supplies that
#: partition for ERP records.
DETERMINISTIC_CANDIDATES = (
    "normal_unknown",
    "line_noise",
    "channel_corruption",
    "emg",
)

HINT_CANDIDATES = (
    "low_snr",
    "drift",
    "spike",
    "eog",
)

#: Ten-dimensional ERP detector: six deterministic states (slow drift and a
#: blink criterion join the four above) and two hints.
ERP_DETERMINISTIC_CANDIDATES = (
    "normal_unknown",
    "line_noise",
    "channel_corruption",
    "emg",
    "drift",
    "eog",
)

ERP_HINT_CANDIDATES = (
    "low_snr",
    "spike",
)

#: Full names used in the main text, for logs and tables.
STATE_LABELS = {
    "normal_unknown": "normal/unknown",
    "line_noise": "line-noise contamination",
    "channel_corruption": "channel corruption",
    "emg": "EMG artifact",
    "low_snr": "low SNR",
    "drift": "baseline drift",
    "spike": "transient spike",
    "eog": "EOG artifact",
}


@dataclass(frozen=True)
class StateSet:
    """A set of states attached to one record.

    A record may hit several states at once.  An empty set means
    ``normal_unknown`` by convention.
    """

    states: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = set(self.states) - set(STATES)
        if unknown:
            raise ValueError(f"unknown states: {sorted(unknown)}")

    @classmethod
    def from_iterable(cls, states: Iterable[str]) -> "StateSet":
        return cls(frozenset(states))

    @classmethod
    def normal(cls) -> "StateSet":
        return cls(frozenset({"normal_unknown"}))

    def is_normal(self) -> bool:
        return not self.states or self.states == frozenset({"normal_unknown"})

    def key(self) -> str:
        """Stable key for logging a multi-state record."""
        return "|".join(sorted(self.states)) if self.states else "normal_unknown"

    def keys(self) -> List[str]:
        """Single-state keys, one per state in the set.

        Memory and knowledge-graph entries are indexed by single states; a
        record that hits several states is looked up once per state and the
        results are merged by the pipeline.
        """
        return sorted(self.states) if self.states else ["normal_unknown"]


class StateDetector:
    """Interface for a deterministic state detector.

    Implementations receive a fingerprint and return a :class:`StateSet`.
    The framework does not assume any particular detector; the paper's
    thresholds are deliberately not included.
    """

    def __call__(self, fingerprint) -> StateSet:  # pragma: no cover - interface
        raise NotImplementedError


class RuleDetector(StateDetector):
    """Detector assembled from caller-supplied boolean rules.

    ``rules`` maps a state name to a callable ``fingerprint -> bool``.  Only
    states whose rule returns True are included.  The caller owns every rule
    and every threshold.
    """

    def __init__(self, rules: Dict[str, object]):
        self.rules = dict(rules)

    def __call__(self, fingerprint) -> StateSet:
        hit = [name for name, rule in self.rules.items() if bool(rule(fingerprint))]
        return StateSet.from_iterable(hit) if hit else StateSet.normal()
