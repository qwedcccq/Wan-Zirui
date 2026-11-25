"""Calibration memory.

Implements the aggregation used by the paper (Eq. 15) without carrying any of
its measured values:

    g(i, s, a) = mean_a(i, s, a) - mean_none(i, s)
    G(s, a)    = clip(mean_i g(i, s, a), clip_lo, clip_hi)

Rows are first averaged within each subject, then averaged across subjects, so
a subject with several rows does not dominate.  The gain and the subject count
always come from the same set of contexts: the exact context when it has an
entry, otherwise the parallel contexts of the same paradigm.  A memory answer
is never strong below ``min_subjects``.

All thresholds are constructor arguments without defaults.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

FIELDS = ("context", "state_key", "action", "subject", "gain", "source")


@dataclass(frozen=True)
class Entry:
    context: str
    state_key: str
    action: str
    subject: str
    gain: float          # percentage points, subject level
    source: str = ""


@dataclass(frozen=True)
class MemoryAnswer:
    """One aggregated answer: gain, supporting subjects, used contexts."""

    gain: float
    subjects: Tuple[str, ...]
    contexts: Tuple[str, ...]

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)


class CalibrationMemory:
    """Per-subject long-format table with the paper's aggregation."""

    def __init__(
        self,
        clip_lo: float,
        clip_hi: float,
        min_subjects: int,
        strong_gain: float,
        separable_gain: float,
        path: Optional[str] = None,
    ):
        if min_subjects < 1:
            raise ValueError("min_subjects must be >= 1")
        if clip_lo >= clip_hi:
            raise ValueError("clip_lo must be < clip_hi")
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.min_subjects = int(min_subjects)
        self.strong_gain = float(strong_gain)
        self.separable_gain = float(separable_gain)
        self._rows: List[Entry] = []
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------ io
    def load(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                self._rows.append(
                    Entry(rec["context"], rec["state_key"], rec["action"],
                          rec["subject"], float(rec["gain"]), rec.get("source", ""))
                )

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("no path given")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            for r in self._rows:
                writer.writerow(dict(context=r.context, state_key=r.state_key,
                                     action=r.action, subject=r.subject,
                                     gain=r.gain, source=r.source))
        return path

    # ------------------------------------------------------------ contexts
    @staticmethod
    def _context_parts(context: str) -> Tuple[str, str, str]:
        parts = context.split("|")
        while len(parts) < 3:
            parts.append("")
        return parts[0], parts[1], parts[2]

    def parallel_contexts(self, context: str) -> List[str]:
        """Contexts under the same paradigm form parallel entry groups."""
        paradigm = self._context_parts(context)[0]
        seen = {r.context for r in self._rows}
        return sorted(c for c in seen if c != context
                      and self._context_parts(c)[0] == paradigm)

    # ------------------------------------------------------------ aggregate
    def _aggregate(
        self, contexts: Tuple[str, ...], state_key: str, action: str
    ) -> Optional[MemoryAnswer]:
        by_subject: Dict[str, List[float]] = {}
        used = set()
        for r in self._rows:
            if (r.context in contexts and r.state_key == state_key
                    and r.action == action):
                by_subject.setdefault(r.subject, []).append(r.gain)
                used.add(r.context)
        if not by_subject:
            return None
        subject_means = [sum(v) / len(v) for v in by_subject.values()]
        mean = sum(subject_means) / len(subject_means)
        gain = max(self.clip_lo, min(self.clip_hi, mean))
        return MemoryAnswer(gain, tuple(sorted(by_subject)), tuple(sorted(used)))

    def gain(self, context: str, state_key: str, action: str) -> Optional[float]:
        """G(s, a) for the exact context (Eq. 15 aggregation)."""
        answer = self.answer(context, state_key, action)
        return answer.gain if answer else None

    def answer(
        self, context: str, state_key: str, action: str
    ) -> Optional[MemoryAnswer]:
        """Gain and supporting subjects; exact context first, then parallel."""
        answer = self._aggregate((context,), state_key, action)
        if answer is not None:
            return answer
        return self._aggregate(tuple(self.parallel_contexts(context)), state_key, action)

    def answer_any(
        self, context: str, state_keys: List[str], action: str
    ) -> Optional[MemoryAnswer]:
        """Best answer over a state set (entries are indexed per single state)."""
        answers = [self.answer(context, s, action) for s in state_keys]
        answers = [a for a in answers if a is not None]
        if not answers:
            return None
        return max(answers, key=lambda a: a.gain)

    def ranked(self, context: str, state_key: str) -> List[Tuple[str, float]]:
        actions = sorted({r.action for r in self._rows
                          if r.context == context and r.state_key == state_key})
        scored = [(a, self.gain(context, state_key, a)) for a in actions]
        scored = [(a, g) for a, g in scored if g is not None]
        return sorted(scored, key=lambda t: t[1], reverse=True)

    def best(self, context: str, state_key: str) -> Optional[Tuple[str, float]]:
        rows = self.ranked(context, state_key)
        return rows[0] if rows else None

    def n_subjects(self, context: str, state_key: str, action: str) -> int:
        answer = self.answer(context, state_key, action)
        return answer.n_subjects if answer else 0

    def has_strong_evidence(
        self, context: str, state_key: str, action: str
    ) -> bool:
        """Entry exists, meets the sample-size rule, and reaches strong gain."""
        answer = self.answer(context, state_key, action)
        if answer is None:
            return False
        return (answer.n_subjects >= self.min_subjects
                and answer.gain >= self.strong_gain)

    # ------------------------------------------------------------ write back
    def write_back(
        self,
        context: str,
        state_key: str,
        action: str,
        subject: str,
        gain: float,
        source: str = "",
    ) -> Entry:
        row = Entry(context, state_key, action, subject, float(gain), source)
        self._rows.append(row)
        return row

    def provenance(
        self, context: str, state_keys: List[str], action: str
    ) -> Dict[str, object]:
        """Provenance for the answer actually used (exact or parallel)."""
        answers = [self.answer(context, s, action) for s in state_keys]
        answers = [a for a in answers if a is not None]
        if not answers:
            return {
                "context": context,
                "state_keys": list(state_keys),
                "action": action,
                "subjects": [],
                "n_subjects": 0,
                "contexts": [],
                "low_confidence": True,
            }
        answer = max(answers, key=lambda a: a.gain)
        return {
            "context": context,
            "state_keys": list(state_keys),
            "action": action,
            "subjects": list(answer.subjects),
            "n_subjects": answer.n_subjects,
            "contexts": list(answer.contexts),
            "low_confidence": answer.n_subjects < self.min_subjects,
        }

    def __len__(self) -> int:
        return len(self._rows)
