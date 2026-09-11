# -*- coding: utf-8 -*-
"""Consensus helpers for source-grounded, scene-local character entities."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


def speaker_key(value: str) -> str:
    """Normalize harmless typography without guessing semantic aliases."""
    value = str(value or "").strip().casefold()
    return re.sub(r"[\s\u00b7\u2022\u30fb.\uff0e_\u2014-]+", "", value)


class _UnionFind:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


@dataclass
class LocalEntityConsensus:
    """Two-of-three local entity links plus per-dialogue assignment support."""

    same_speaker: Callable[[str, str], bool]
    is_generic_speaker: Callable[[str], bool]
    canonicalize_speaker: Callable[[str], tuple[str, str]]
    root_by_key: dict[str, str] = field(default_factory=dict)
    labels_by_root: dict[str, set[str]] = field(default_factory=dict)
    canonical_votes_by_root: dict[str, Counter[str]] = field(default_factory=dict)
    presence_agents_by_root: dict[str, set[str]] = field(default_factory=dict)
    assignment_agents: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    citations_by_root: dict[str, set[int]] = field(default_factory=dict)
    completed_agents: set[str] = field(default_factory=set)

    @classmethod
    def empty(
        cls,
        *,
        same_speaker: Callable[[str, str], bool],
        is_generic_speaker: Callable[[str], bool],
        canonicalize_speaker: Callable[[str], tuple[str, str]],
    ) -> "LocalEntityConsensus":
        return cls(
            same_speaker=same_speaker,
            is_generic_speaker=is_generic_speaker,
            canonicalize_speaker=canonicalize_speaker,
        )

    def _root(self, label: str) -> str:
        key = speaker_key(label)
        return self.root_by_key.get(key, key)

    def same_entity(self, left: str, right: str) -> bool:
        if self.same_speaker(str(left), str(right)):
            return True
        left_key = speaker_key(left)
        right_key = speaker_key(right)
        if not left_key or not right_key:
            return False
        return (
            left_key in self.root_by_key
            and right_key in self.root_by_key
            and self._root(left) == self._root(right)
        )

    def labels_for(self, label: str) -> list[str]:
        return sorted(
            self.labels_by_root.get(self._root(label), {str(label)}),
            key=lambda item: (-len(item), item),
        )

    def presence_support(self, label: str) -> int:
        return len(self.presence_agents_by_root.get(self._root(label), set()))

    def citations_for(self, label: str) -> set[int]:
        return set(self.citations_by_root.get(self._root(label), set()))

    @property
    def agent_count(self) -> int:
        return len(self.completed_agents)

    def assignment_support(self, dialogue_id: str, label: str) -> int:
        roots = self.assignment_agents.get(str(dialogue_id), {})
        return len(roots.get(self._root(label), set()))

    def competing_named_support(self, dialogue_id: str, label: str) -> int:
        target_root = self._root(label)
        best = 0
        for root, agents in self.assignment_agents.get(str(dialogue_id), {}).items():
            if root == target_root:
                continue
            if not self.presence_agents_by_root.get(root):
                continue
            labels = self.labels_by_root.get(root, set())
            if any(not self.is_generic_speaker(item) for item in labels):
                best = max(best, len(agents))
        return best

    def canonical_for(self, label: str, *, baseline: str = "") -> tuple[str, str]:
        root = self._root(label)
        if baseline and self.same_entity(label, baseline):
            return str(baseline), "local-entity-preserved-baseline"

        votes = self.canonical_votes_by_root.get(root, Counter())
        candidates = [item for item, _count in votes.most_common()]
        candidates.extend(
            item
            for item in self.labels_by_root.get(root, set())
            if item not in candidates
        )
        verified = []
        for candidate in candidates:
            canonical, note = self.canonicalize_speaker(candidate)
            if note:
                verified.append((votes.get(candidate, 0), canonical, note))
        if verified:
            _count, canonical, note = max(verified, key=lambda item: (item[0], len(item[1])))
            return canonical, note
        if candidates:
            winner = max(
                candidates,
                key=lambda item: (
                    not self.is_generic_speaker(item),
                    votes.get(item, 0),
                    len(item),
                    item,
                ),
            )
            return winner, "local-entity-canonical-label"
        return str(label), ""

    def summary(self) -> dict[str, Any]:
        return {
            "agents": len(self.completed_agents),
            "entities": [
                {
                    "labels": sorted(labels),
                    "presence_support": len(self.presence_agents_by_root.get(root, set())),
                    "citations": sorted(self.citations_by_root.get(root, set())),
                }
                for root, labels in sorted(self.labels_by_root.items())
            ],
        }


def build_local_entity_consensus(
    records: list[dict[str, Any]],
    *,
    same_speaker: Callable[[str, str], bool],
    is_generic_speaker: Callable[[str], bool],
    canonicalize_speaker: Callable[[str], tuple[str, str]],
    min_link_support: int = 2,
) -> LocalEntityConsensus:
    active = [record for record in records if not record.get("abstained")]
    if not active:
        return LocalEntityConsensus.empty(
            same_speaker=same_speaker,
            is_generic_speaker=is_generic_speaker,
            canonicalize_speaker=canonicalize_speaker,
        )

    display_by_key: dict[str, str] = {}
    pair_agents: dict[tuple[str, str], set[str]] = defaultdict(set)
    canonical_agents: dict[tuple[str, str], set[str]] = defaultdict(set)
    presence_agents: dict[str, set[str]] = defaultdict(set)
    citation_values: dict[str, set[int]] = defaultdict(set)
    assignment_agents: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    completed_agents: set[str] = set()

    for record in active:
        agent = str(record.get("agent") or "")
        if not agent:
            continue
        completed_agents.add(agent)
        graph = record.get("graph") or record
        assignments_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for assignment in graph.get("assignments") or []:
            assignments_by_entity[str(assignment.get("entity_id") or "")].append(assignment)

        for entity in graph.get("entities") or []:
            entity_id = str(entity.get("entity_id") or "")
            canonical = str(entity.get("canonical_label") or "").strip()
            labels = {
                str(item).strip()
                for item in [canonical, *(entity.get("mentions") or [])]
                if str(item).strip()
            }
            labels.update(
                str(item.get("speaker") or "").strip()
                for item in assignments_by_entity.get(entity_id, [])
                if str(item.get("speaker") or "").strip()
            )
            keys = sorted({speaker_key(label) for label in labels if speaker_key(label)})
            for label in labels:
                key = speaker_key(label)
                if key:
                    display_by_key.setdefault(key, label)
                    presence_agents[key].add(agent)
                    citation_values[key].update(
                        int(value)
                        for value in entity.get("citations") or []
                        if str(value).isdigit()
                    )
                    if canonical:
                        canonical_agents[(key, canonical)].add(agent)
            for offset, left in enumerate(keys):
                for right in keys[offset + 1 :]:
                    pair_agents[(left, right)].add(agent)

        for assignment in graph.get("assignments") or []:
            dialogue_id = str(assignment.get("id") or "")
            speaker = str(assignment.get("speaker") or "").strip()
            key = speaker_key(speaker)
            if not dialogue_id or not key:
                continue
            display_by_key.setdefault(key, speaker)
            assignment_agents[dialogue_id][key].add(agent)

    keys = set(display_by_key)
    union = _UnionFind(keys)
    key_list = sorted(keys)
    for offset, left in enumerate(key_list):
        for right in key_list[offset + 1 :]:
            if same_speaker(display_by_key[left], display_by_key[right]):
                union.union(left, right)
    for (left, right), agents in pair_agents.items():
        if len(agents) >= max(1, int(min_link_support)):
            union.union(left, right)

    root_by_key = {key: union.find(key) for key in keys}
    labels_by_root: dict[str, set[str]] = defaultdict(set)
    canonical_votes_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    presence_by_root: dict[str, set[str]] = defaultdict(set)
    citations_by_root: dict[str, set[int]] = defaultdict(set)
    for key, label in display_by_key.items():
        root = root_by_key[key]
        labels_by_root[root].add(label)
        presence_by_root[root].update(presence_agents.get(key, set()))
        citations_by_root[root].update(citation_values.get(key, set()))
    for (key, canonical), agents in canonical_agents.items():
        canonical_votes_by_root[root_by_key[key]][canonical] += len(agents)

    assignments_by_root: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for dialogue_id, by_key in assignment_agents.items():
        for key, agents in by_key.items():
            assignments_by_root[dialogue_id][root_by_key.get(key, key)].update(agents)

    return LocalEntityConsensus(
        same_speaker=same_speaker,
        is_generic_speaker=is_generic_speaker,
        canonicalize_speaker=canonicalize_speaker,
        root_by_key=root_by_key,
        labels_by_root=dict(labels_by_root),
        canonical_votes_by_root=dict(canonical_votes_by_root),
        presence_agents_by_root=dict(presence_by_root),
        assignment_agents={
            dialogue_id: dict(by_root)
            for dialogue_id, by_root in assignments_by_root.items()
        },
        citations_by_root=dict(citations_by_root),
        completed_agents=completed_agents,
    )
