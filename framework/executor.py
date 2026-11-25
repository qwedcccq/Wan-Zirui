"""Action registry and execution interface.

The framework separates *selection* from *execution*: the decision chain
returns an action identifier, and a fixed, pre-registered function executes
it.  This module provides only the registry and the calling convention.  No
preprocessing method is implemented here; the reader registers their own.

Two stages are supported, matching the paper:

- preprocessing actions: ``fn(record, fs) -> record``;
- feature-stage actions: ``fn(epochs, labels) -> score`` (for example a
  spatial filter fitted on training folds and scored by cross-validation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: ``fn(record, fs) -> record`` for a preprocessing-stage action.
ActionFn = Callable[[np.ndarray, float], np.ndarray]

#: ``fn(epochs, labels) -> score`` for a feature-stage action.
FeatureActionFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    stage: str                                   # "preprocessing" or "feature"
    fn: Optional[Any] = None                     # None means decision-only
    description: str = ""
    paradigm: str = ""                           # e.g. "mi", "ssvep"
    slot: str = ""                               # interchangeable group
    prerequisites: Tuple[str, ...] = field(default_factory=tuple)
    exclusions: Tuple[str, ...] = field(default_factory=tuple)


class ActionRegistry:
    """Registry of actions with metadata and an executable set.

    A decision-only entry (``fn=None``) is listed in the candidate space but
    excluded from the executable set, mirroring the paper's rule that
    adoption cannot change the execution set.
    """

    def __init__(self) -> None:
        self._specs: Dict[str, ActionSpec] = {}

    def register(
        self,
        name: str,
        fn: Optional[Any],
        stage: str = "preprocessing",
        description: str = "",
        paradigm: str = "",
        slot: str = "",
        prerequisites: Iterable[str] = (),
        exclusions: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        if name in self._specs and not replace:
            raise ValueError(f"action already registered: {name}")
        if stage not in ("preprocessing", "feature"):
            raise ValueError(f"unknown stage: {stage}")
        self._specs[name] = ActionSpec(
            name=name, stage=stage, fn=fn, description=description,
            paradigm=paradigm, slot=slot,
            prerequisites=tuple(prerequisites), exclusions=tuple(exclusions),
        )

    def register_many(self, names: Iterable[str], fn: Optional[Any] = None,
                      stage: str = "preprocessing", description: str = "",
                      paradigm: str = "", slot: str = "",
                      prerequisites: Iterable[str] = (),
                      exclusions: Iterable[str] = (),
                      replace: bool = False) -> None:
        for name in names:
            self.register(name, fn, stage, description, paradigm, slot,
                          prerequisites, exclusions, replace)

    def executable(self) -> set:
        return {s.name for s in self._specs.values() if s.fn is not None}

    def candidates(self) -> List[str]:
        return sorted(self._specs)

    def by(self, paradigm: Optional[str] = None, slot: Optional[str] = None,
           stage: Optional[str] = None) -> List[str]:
        """Filter registered actions by paradigm, slot, and stage.

        A paradigm filter matches exactly; an action registered with
        ``paradigm="*"`` is treated as paradigm-general.  Unlabelled actions
        are not returned for a specific paradigm.
        """
        out = []
        for spec in self._specs.values():
            if paradigm is not None and spec.paradigm not in (paradigm, "*"):
                continue
            if slot is not None and spec.slot != slot:
                continue
            if stage is not None and spec.stage != stage:
                continue
            out.append(spec.name)
        return sorted(out)

    def get(self, name: str) -> ActionSpec:
        return self._specs[name]

    def execute(self, name: str, record: np.ndarray, fs: float) -> np.ndarray:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unregistered action: {name}")
        if spec.fn is None:
            raise RuntimeError(f"action '{name}' is decision-only (no tool)")
        if spec.stage != "preprocessing":
            raise RuntimeError(f"action '{name}' is not a preprocessing action")
        return np.asarray(spec.fn(record, fs), dtype=float)

    def execute_feature(
        self, name: str, epochs: np.ndarray, labels: np.ndarray
    ) -> float:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unregistered action: {name}")
        if spec.fn is None:
            raise RuntimeError(f"action '{name}' is decision-only (no tool)")
        if spec.stage != "feature":
            raise RuntimeError(f"action '{name}' is not a feature-stage action")
        return float(spec.fn(epochs, labels))
