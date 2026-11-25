"""Knowledge-graph interface.

The graph maps data states to candidate actions and a mechanistic order.
The paper's nodes, edges, and weights are prior knowledge and are not
included.  The reader supplies their own graph.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Protocol, Tuple


class KnowledgeGraph(Protocol):
    """``state -> candidates`` and ``state, candidates -> order``."""

    def candidates(self, state_key: str) -> List[str]:  # pragma: no cover
        ...

    def rank(self, state_key: str, candidates: Iterable[str]) -> List[str]:  # pragma: no cover
        ...


class DictKnowledgeGraph:
    """Minimal weighted graph backed by a caller-supplied mapping.

    ``edges`` maps a state key to ``(action, weight)`` pairs.  ``rank`` orders
    candidates by weight (highest first); ties keep insertion order.  No
    default edges are provided.
    """

    def __init__(self, edges: Dict[str, Iterable[Tuple[str, float]]] | None = None):
        self.edges: Dict[str, List[Tuple[str, float]]] = {}
        for key, pairs in (edges or {}).items():
            self.edges[key] = [(a, float(w)) for a, w in pairs]

    def candidates(self, state_key: str) -> List[str]:
        return [a for a, _ in self.edges.get(state_key, [])]

    def weight(self, state_key: str, action: str) -> float:
        for a, w in self.edges.get(state_key, []):
            if a == action:
                return w
        return 0.0

    def rank(self, state_key: str, candidates: Iterable[str]) -> List[str]:
        position = {a: i for i, (a, _) in enumerate(self.edges.get(state_key, []))}
        return sorted(
            candidates,
            key=lambda c: (-self.weight(state_key, c), position.get(c, len(position))),
        )

    def add_edge(self, state_key: str, action: str, weight: float = 0.0) -> None:
        """Insert or update an edge, keeping the list ordered by weight."""
        pairs = self.edges.setdefault(state_key, [])
        for i, (a, _) in enumerate(pairs):
            if a == action:
                pairs[i] = (a, float(weight))
                break
        else:
            pairs.append((action, float(weight)))
        pairs.sort(key=lambda t: t[1], reverse=True)

    def candidates_multi(self, state_keys: Iterable[str]) -> List[str]:
        """Union of candidates over a state set, ordered by summed weight."""
        totals: Dict[str, float] = {}
        first: Dict[str, int] = {}
        seq = 0
        for key in state_keys:
            for action, weight in self.edges.get(key, []):
                totals[action] = totals.get(action, 0.0) + weight
                if action not in first:
                    first[action] = seq
                    seq += 1
        return sorted(totals, key=lambda a: (-totals[a], first[a]))
