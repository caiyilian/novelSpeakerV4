"""Scene-level, evidence-gated speaker attribution orchestration.

The runtime-specific API and logging functions are injected through hooks so
this module can focus on scene maps, evidence investigations, and conservative
selection without importing the monolithic CLI module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from scene_sequence import (
    assignment_for_dialogue,
    build_scene_packet,
    build_scene_segments,
    find_scene_for_dialogue,
    format_scene_packet,
    normalize_scene_assignments,
)


_SPEECH_VERB_PATTERN = (
    r"(?:说(?!不(?:出|上|下|成|定|清|明|来))(?:道|着|完|了)?|问(?:道)?|回答|答道|喊(?:道)?|叫(?:道)?|开口|"
    r"低语|嘀咕|喃喃|叹(?:道)?|笑(?:道)?|补充|表示|"
    r"搭腔|搭话|接话|插话|回应|应声)"
)
_DIALOGUE_CONTACT_VERBS = {"搭腔", "搭话", "接话", "插话", "回应", "应声"}
_NONPERSON_CONTAINER_RE = re.compile(
    r"(?:没有说出口|没说出口|未说出口|并未说出口|不曾说出口|"
    r"没有开口|并未开口|省略(?:掉)?(?:了)?(?:这|那)?句[^。！？]{0,16}话(?:语)?|"
    r"心(?:里|中|底)(?:暗自)?(?:想|想着|想到)|内心(?:暗自)?(?:想|想着|想到)|"
    r"脑海(?:中|里)[^。！？]{0,12}(?:浮现|响起))"
)
_PARTIAL_SPEECH_CONTAINER_RE = re.compile(
    r"(?:说到一半|话(?:才|刚)?说到一半|刚(?:要)?开口|才刚开口|"
    r"话(?:才|刚)?出口|刚说出|已经说出|脱口而出)"
)
_PERSON_VOCALIZATION_PATTERN = (
    r"(?:发出[^。！？]{0,24}(?:鼾声|笑声|哭声|呻吟声|叫声|叹息声|咳嗽声|声音)|"
    r"打(?:起)?鼾|发笑|笑出声|哭出声|呻吟|呜咽|抽泣|咳嗽|惊呼|喘息)"
)
_CONVEYED_QUOTE_PATTERN = re.compile(
    r"(?:仿佛|好像|像是|似乎)[^。！？；]{0,32}(?:在)?(?:说|问|回答|喊|表示)"
)
_CONVEYED_EXPRESSION_PATTERN = re.compile(
    r"(?:一副|满脸|露出|显出|带着)?[^。！？；]{0,16}[「『“][^」』”]{1,80}[」』”]"
    r"(?:的)?(?:模样|表情|神情|样子|眼神)"
)


SCENE_MAP_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_scene_map",
        "description": "Submit a complete chronological speaker map for every displayed scene turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn_id": {"type": "string"},
                            "line": {"type": "integer"},
                            "speaker": {"type": "string"},
                            "quote_type": {
                                "type": "string",
                                "enum": [
                                    "direct_speech", "group_speech", "embedded_quote",
                                    "thought_or_narration", "sound_or_text", "unclear",
                                ],
                            },
                            "evidence_basis": {
                                "type": "string",
                                "enum": [
                                    "explicit_attribution", "speaker_action", "identity_alias",
                                    "quote_type", "sequence_constraint", "inference", "unknown",
                                ],
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "citations": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "turn_id", "line", "speaker", "quote_type",
                            "evidence_basis", "confidence", "citations", "reason",
                        ],
                    },
                },
                "scene_notes": {"type": "string"},
            },
            "required": ["assignments", "scene_notes"],
        },
    },
}


TARGET_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_target_evidence",
        "description": "Submit one evidence-based target attribution or an explicit unresolved result.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["resolved", "unresolved"]},
                "speaker": {"type": "string"},
                "voice_kind": {"type": "string", "enum": ["person", "non_person", "unclear"]},
                "quote_type": {
                    "type": "string",
                    "enum": [
                        "direct_speech", "group_speech", "embedded_quote",
                        "thought_or_narration", "sound_or_text", "unclear",
                    ],
                },
                "evidence_basis": {
                    "type": "string",
                    "enum": [
                        "explicit_attribution", "speaker_action", "identity_alias",
                        "quote_type", "sequence_constraint", "inference", "unknown",
                    ],
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "citations": {"type": "array", "items": {"type": "integer"}},
                "reason": {"type": "string"},
                "rejected_candidates": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "status", "speaker", "voice_kind", "quote_type", "evidence_basis",
                "confidence", "citations", "reason", "rejected_candidates",
            ],
        },
    },
}


SCENE_VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_scene_verdict",
        "description": "Submit the final evidence comparison without using candidate vote counts.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["resolved", "unresolved"]},
                "speaker": {"type": "string"},
                "voice_kind": {"type": "string", "enum": ["person", "non_person", "unclear"]},
                "quote_type": {
                    "type": "string",
                    "enum": [
                        "direct_speech", "group_speech", "embedded_quote",
                        "thought_or_narration", "sound_or_text", "unclear",
                    ],
                },
                "evidence_level": {
                    "type": "string",
                    "enum": [
                        "direct", "identity", "quote_type",
                        "corroborated_sequence", "weak", "insufficient",
                    ],
                },
                "evidence_basis": {
                    "type": "string",
                    "enum": [
                        "explicit_attribution", "speaker_action", "identity_alias",
                        "quote_type", "sequence_constraint", "inference", "unknown",
                    ],
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "citations": {"type": "array", "items": {"type": "integer"}},
                "reason": {"type": "string"},
                "contradicted_candidates": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "status", "speaker", "voice_kind", "quote_type", "evidence_level", "evidence_basis",
                "confidence", "citations", "reason", "contradicted_candidates",
            ],
        },
    },
}


BOUNDARY_BRIEF_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_boundary_brief",
        "description": "Submit the strongest evidence and weaknesses for one assigned target-speaker hypothesis.",
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string", "enum": ["A", "B"]},
                "supporting_citations": {"type": "array", "items": {"type": "integer"}},
                "contradicting_citations": {"type": "array", "items": {"type": "integer"}},
                "strongest_case": {"type": "string"},
                "weakest_assumption": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": [
                "hypothesis_id", "supporting_citations", "contradicting_citations",
                "strongest_case", "weakest_assumption", "confidence",
            ],
        },
    },
}


BOUNDARY_JURY_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_boundary_verdict",
        "description": "Choose between two target-speaker hypotheses after explicit boundary comparison.",
        "parameters": {
            "type": "object",
            "properties": {
                "winner_id": {"type": "string", "enum": ["A", "B", "unresolved"]},
                "target_to_previous": {
                    "type": "string",
                    "enum": ["same_speaker", "different_speaker", "unclear"],
                },
                "target_to_next": {
                    "type": "string",
                    "enum": ["same_speaker", "different_speaker", "unclear"],
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "citations": {"type": "array", "items": {"type": "integer"}},
                "decisive_evidence": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": [
                "winner_id", "target_to_previous", "target_to_next", "confidence",
                "citations", "decisive_evidence", "reason",
            ],
        },
    },
}


@dataclass
class SceneQualityRuntime:
    call_model: Callable[..., tuple[str, int, int, list[dict[str, Any]]]]
    parse_tool_arguments: Callable[[str, Any], tuple[dict[str, Any], str]]
    normalize_tool_call: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    log_agent: Callable[..., None]
    trace_tool_result: Callable[..., None]
    temp_log_event: Callable[..., None]
    read_novel_lines: Callable[[int, int], str]
    search_novel: Callable[[str, int], str]
    validate_speaker: Callable[[str], str | None]
    is_valid_speaker: Callable[[str], bool]
    is_generic_speaker: Callable[[str], bool]
    same_speaker: Callable[[str, str], bool]
    estimate_messages: Callable[[list[dict[str, Any]]], int]
    read_tool: dict[str, Any]
    search_tool: dict[str, Any]
    non_person_label: str
    request_timeout: int = 180
    retries: int = 2
    deep_tool_rounds: int = 12
    deep_token_budget: int = 200000
    context_limit: int = 262144
    max_output_tokens: int = 16384


class SceneSequenceAudit:
    """Generate complete scene maps and evidence-gate target corrections."""

    def __init__(
        self,
        novel_lines: list[str],
        dialogue_list: list[tuple[int, str]],
        runtime: SceneQualityRuntime,
        *,
        max_scene_turns: int = 6,
        max_scene_raw_span: int = 180,
        scene_context_padding: int = 80,
        allow_model_consensus_override: bool = False,
    ):
        self.novel_lines = novel_lines
        self.dialogue_list = dialogue_list
        self.runtime = runtime
        self.segments = build_scene_segments(
            novel_lines,
            dialogue_list,
            max_turns=max_scene_turns,
            max_raw_span=max_scene_raw_span,
            raw_padding=scene_context_padding,
        )
        self.allow_model_consensus_override = bool(allow_model_consensus_override)
        self.scene_cache: dict[str, dict[str, Any]] = {}
        self.model_call_count = 0

    def decide(
        self,
        dialogue_index: int,
        line_num: int,
        dialogue: str,
        baseline: str,
        baseline_evidence: dict[str, Any],
        baseline_reason: str,
        local_evidence: dict[str, Any],
        round_log: dict[str, Any],
    ) -> tuple[str, str, str, int, int, dict[str, Any]]:
        calls_before = self.model_call_count
        segment = find_scene_for_dialogue(self.segments, dialogue_index)
        scene_id = segment["scene_id"]
        cache_hit = scene_id in self.scene_cache
        total_pec = 0
        total_ec = 0

        if cache_hit:
            scene_solution = self.scene_cache[scene_id]
        else:
            packet = build_scene_packet(self.novel_lines, self.dialogue_list, segment)
            scene_solution, pec, ec = self._solve_scene(packet, round_log)
            scene_solution["source_trace_id"] = round_log.get("trace_id")
            self.scene_cache[scene_id] = scene_solution
            total_pec += pec
            total_ec += ec

        packet = scene_solution["packet"]
        scene_cards = self._target_scene_cards(scene_solution, dialogue_index)
        evidence_cases = self._build_evidence_cases(
            baseline,
            baseline_evidence,
            baseline_reason,
            scene_cards,
            local_evidence,
            line_num,
            include_baseline=False,
        )
        neutral_cases = self._neutralize_cases(evidence_cases, line_num)
        neutral_sequences = self._neutralize_scene_sequences(scene_solution, line_num)
        target_packet_text = format_scene_packet(
            packet,
            target_dialogue_index=dialogue_index,
            max_raw_chars=24000,
        )
        target_packet_text += "\n\n" + self._format_boundary_constraints(packet, dialogue_index)
        baseline_scope_audit = self._baseline_scope_conflict(
            baseline,
            baseline_evidence,
            baseline_reason,
            packet,
            dialogue_index,
            local_evidence,
        )

        investigations = []
        investigator_specs = [
            (
                "TargetBindingInvestigator",
                "Resolve exact grammatical binding for the target. Prioritize speech verbs, action subjects, "
                "quote containers, and post-quote attribution. A narrative attribution immediately after a quote "
                "normally belongs to that preceding quote, not the next quote. Expand the raw range when binding "
                "is outside the scene.",
                neutral_cases,
                neutral_sequences,
            ),
            (
                "TargetOpenWorldInvestigator",
                "Determine the target speaker independently from raw text. You are intentionally given no candidate "
                "labels or proposed sequences. Expand context, identify all participants, and search for identity or "
                "alias evidence when the local text uses a temporary role. Do not imitate an assumed baseline.",
                [],
                [],
            ),
            (
                "TargetSequenceBoundaryInvestigator",
                "Compare the anonymous complete-scene sequences against raw chronology. Resolve whether the target "
                "continues the previous speaker or switches, and whether a nearby attribution points backward to the "
                "preceding quote or forward to the target. Reject unexplained alternation.",
                [],
                neutral_sequences,
            ),
            (
                "TargetIdentityContainerInvestigator",
                "Independently classify the exact quote container and speaker identity. Distinguish spoken words, "
                "started-but-interrupted speech, quoted meaning conveyed by a named person's expression, hypothetical "
                "thought, displayed text, and object sounds. If a person is described by a temporary role, use "
                "search_novel to prove or reject a named identity before submitting.",
                [],
                [],
            ),
            (
                "TargetDelayedAttributionInvestigator",
                "Scan from the target through the end of the raw packet for delayed attribution of an entire block of "
                "consecutive quotes. Later narration may reveal that several preceding lines were overheard from "
                "passersby, a crowd, workers, customers, or another unnamed human group. Such human speech is person "
                "voice: use a stable source role such as 路人 for an unidentified background individual or 人群 for a "
                "collective chorus, never 非人物发声 merely because no personal name is known.",
                [],
                [],
            ),
        ]
        for agent_name, instruction, agent_cases, agent_sequences in investigator_specs:
            report, pec, ec = self._run_investigator(
                agent_name,
                instruction,
                target_packet_text,
                agent_cases,
                agent_sequences,
                packet,
                round_log,
            )
            report = dict(report)
            report["agent"] = agent_name
            investigations.append(report)
            total_pec += pec
            total_ec += ec

        baseline_hypothesis = self.runtime.validate_speaker(baseline) or baseline or "?"
        falsifier_instruction = (
            f"The provisional target-speaker hypothesis is {baseline_hypothesis}. This label is fallible and is "
            "shown only so you can search for a missing candidate; it carries no authority. Act as a baseline "
            "falsifier and candidate generator. Reconstruct the previous, target, and next turns twice: first under "
            "the provisional hypothesis, then under the strongest different source speaker actually present in raw "
            "text. Pay special attention to a question followed by an answer, consecutive quotes split without "
            "narration, a listener action that merely prompts a reply, and non-colon narration after the previous "
            "quote that may already be consumed by that quote. Actively expand context when the local packet begins "
            "mid-exchange. Submit the strongest alternative when it survives raw-text contradiction; otherwise "
            "submit the provisional speaker. Never invent a character and never decide from style, topic, occupation, "
            "species, or mechanical alternation. Your report is candidate evidence only and cannot directly override "
            "the final label."
        )
        report, pec, ec = self._run_investigator(
            "TargetBaselineFalsifier",
            falsifier_instruction,
            target_packet_text,
            [],
            neutral_sequences,
            packet,
            round_log,
        )
        report = dict(report)
        report["agent"] = "TargetBaselineFalsifier"
        investigations.append(report)
        total_pec += pec
        total_ec += ec

        verdict, pec, ec = self._run_verdict(
            target_packet_text,
            neutral_cases,
            neutral_sequences,
            investigations,
            packet,
            round_log,
        )
        total_pec += pec
        total_ec += ec

        if (
            baseline_scope_audit.get("conflict")
            and _DIALOGUE_CONTACT_VERBS.intersection(
                baseline_scope_audit.get("consumed_verbs") or []
            )
        ):
            consumed_lines = ", ".join(
                f"L{line}" for line in baseline_scope_audit.get("consumed_lines") or []
            )
            baseline_clean = self.runtime.validate_speaker(baseline) or baseline or "?"
            contact_specs = [
                (
                    "TargetContactScopeRepairForward",
                    "Reconstruct forward from the preceding quote through TARGET and the following quote. Track who "
                    "holds the conversational floor and require raw evidence for every switch.",
                ),
                (
                    "TargetContactScopeRepairReverse",
                    "Reconstruct backward from the first reliable attribution or action after TARGET. Test whether "
                    "TARGET is a reply to the consumed contact utterance or an independently supported continuation.",
                ),
                (
                    "TargetContactScopeRepairDialogueAct",
                    "Map address, greeting, question-answer, thanks-reciprocation, correction, and elaboration roles "
                    "across the previous, target, and next turns without using personality or occupation.",
                ),
            ]
            for agent_name, perspective in contact_specs:
                instruction = (
                    f"A deterministic quote parser has established that {consumed_lines} contains a non-colon "
                    "contact-speech attribution consumed by the PRECEDING quote. It cannot be used as attribution "
                    f"for TARGET. The provisional label {baseline_clean} relied on that consumed line and is not an "
                    "authority. The same person may still continue, but only independent target evidence can prove "
                    f"that. {perspective} Submit unresolved when neither hypothesis is grounded."
                )
                report, repair_pec, repair_ec = self._run_investigator(
                    agent_name,
                    instruction,
                    target_packet_text,
                    [],
                    [],
                    packet,
                    round_log,
                )
                report = dict(report)
                report["agent"] = agent_name
                investigations.append(report)
                total_pec += repair_pec
                total_ec += repair_ec

        contrastive_boundary, pec, ec = self._run_contrastive_boundary(
            baseline,
            scene_cards,
            investigations,
            verdict,
            packet,
            dialogue_index,
            target_packet_text,
            round_log,
            neighbor_candidates=self._neighboring_scene_candidates(
                scene_solution,
                dialogue_index,
                baseline,
            ),
        )
        total_pec += pec
        total_ec += ec

        decision = self._select_decision(
            baseline,
            baseline_evidence,
            scene_cards,
            investigations,
            verdict,
            packet,
            dialogue_index,
            local_evidence,
            contrastive_boundary,
            baseline_reason=baseline_reason,
        )
        speaker = decision["speaker"]
        summary = f"{speaker} | {dialogue[:120]}"
        reason = (
            f"scene={scene_id}; source={decision['selection_source']}; "
            f"baseline={baseline}; verdict={verdict.get('speaker') or '?'}; "
            f"gate={decision.get('evidence_gate', 'none')}"
        )
        metadata = {
            "decision": decision,
            "scene": {
                "scene_id": scene_id,
                "start_index": packet["start_index"],
                "end_index": packet["end_index"],
                "raw_start": packet["raw_start"],
                "raw_end": packet["raw_end"],
                "turn_count": len(packet["turns"]),
                "cache_hit": cache_hit,
                "source_trace_id": scene_solution.get("source_trace_id"),
            },
            "scene_maps": scene_solution["maps"] if not cache_hit else [],
            "scene_map_reference": (
                None if not cache_hit else {
                    "scene_id": scene_id,
                    "source_trace_id": scene_solution.get("source_trace_id"),
                }
            ),
            "target_scene_cards": scene_cards,
            "neutral_evidence_cases": neutral_cases,
            "neutral_scene_sequences": neutral_sequences,
            "investigations": investigations,
            "verdict": verdict,
            "contrastive_boundary": contrastive_boundary,
            "model_calls": self.model_call_count - calls_before,
        }
        return speaker, summary, reason, total_pec, total_ec, metadata

    def _solve_scene(
        self,
        packet: dict[str, Any],
        round_log: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int]:
        packet_text = format_scene_packet(packet, max_raw_chars=26000)
        common = (
            "Map EVERY displayed dialogue turn from first to last. Return exactly one assignment for each D id. "
            "A person may speak multiple consecutive turns; never force alternation. A mentioned person or addressee "
            "is not automatically the speaker. Use only supplied raw text. Copy Chinese source names or stable unnamed "
            "roles exactly; use ? when evidence is genuinely insufficient. Use 非人物发声 only when nobody utters "
            "the exact span. An adjacent action subject is not a direct speaker attribution: the action may prompt the "
            "other participant's reply. Mark explicit_attribution only when a speech verb binds that subject. "
            "A named person's laugh, sob, cough, groan, snore, or bodily/vocal sound belongs to that person, not to "
            "非人物发声. Speech that starts and is then interrupted also belongs to its speaker. "
            "A narrative speech attribution immediately after a quote normally binds that preceding quote; do not "
            "silently move it forward to the next quote unless punctuation or wording explicitly introduces the next "
            "utterance. Quoted meaning conveyed by a named person's expression or gesture belongs to that person as "
            "an embedded quote under this annotation schema. "
            "Question-answer, proposal-acceptance, farewell-response, and clarification-explanation pairs are real "
            "sequence evidence when grounded in the supplied text; use them without assuming that every turn alternates. "
            "Always scan narration after the displayed quote block for delayed attribution. Unnamed human passersby, "
            "workers, customers, or crowds are person speakers; copy a stable source role such as 路人 or 人群 rather "
            "than treating unknown identity as 非人物发声. "
            "Every assignment needs a supplied L-number citation and a concise reason. "
            "Call submit_scene_map without prose before the tool call."
        )
        mapper_specs = [
            (
                "SceneStructureMapper",
                "Focus only on grammatical quote binding, speech verbs, action subjects, post-quote attribution, "
                "embedded writing/thought, and object sounds. Do not use topic, persona, occupation, or presumed style.",
            ),
            (
                "SceneChronologyMapper",
                "Construct the complete chronological speaker sequence. For each transition, decide whether raw text "
                "proves a switch, proves continuation, or leaves it uncertain. Consecutive quotes may share a speaker.",
            ),
            (
                "SceneContinuityMapper",
                "Track questions, answers, address forms, interrupted speech, and narrative actions across the full "
                "scene. Treat dialogue flow as supporting evidence only and reject character-style guessing.",
            ),
            (
                "SceneDialogueActMapper",
                "Classify each turn's dialogue act and resolve grounded adjacency pairs such as question-answer, "
                "request-response, correction-acknowledgment, and farewell-response. A proven response relation can "
                "support a speaker switch, but never impose mechanical alternation.",
            ),
            (
                "SceneReverseMapper",
                "Solve the complete sequence backward from the final turn. Bind post-quote narration to the quote "
                "immediately before it, then propagate only transitions supported by dialogue acts or narration. "
                "Use this reverse pass to challenge forward-reading anchoring.",
            ),
        ]
        maps = []
        total_pec = 0
        total_ec = 0
        for agent_name, instruction in mapper_specs:
            def validator(args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
                normalized = normalize_scene_assignments(
                    args.get("assignments"),
                    packet,
                    self._map_speaker,
                )
                return normalized["valid"], normalized

            args, normalized, pec, ec = self._run_required_tool(
                agent_name,
                "complete_scene_mapper",
                instruction + "\n\n" + common,
                packet_text,
                SCENE_MAP_TOOL,
                "submit_scene_map",
                validator,
                round_log,
            )
            total_pec += pec
            total_ec += ec
            maps.append(
                {
                    "agent": agent_name,
                    "valid": bool(normalized and normalized.get("valid")),
                    "assignments": (normalized or {}).get("assignments", {}),
                    "missing": (normalized or {}).get("missing", []),
                    "errors": (normalized or {}).get("errors", []),
                    "scene_notes": str((args or {}).get("scene_notes") or "")[:2000],
                }
            )
        return {"packet": packet, "maps": maps}, total_pec, total_ec

    def _run_required_tool(
        self,
        agent_name: str,
        role: str,
        system_prompt: str,
        user_content: str,
        tool: dict[str, Any],
        expected_name: str,
        validator: Callable[[dict[str, Any]], tuple[bool, Any]],
        round_log: dict[str, Any],
    ) -> tuple[dict[str, Any], Any, int, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        total_pec = 0
        total_ec = 0
        last_args: dict[str, Any] = {}
        last_normalized: Any = None
        last_error = "missing required tool call"

        for attempt in range(1, max(1, self.runtime.retries) + 1):
            try:
                text, pec, ec, tool_calls = self.runtime.call_model(
                    messages,
                    tools=[tool],
                    label=f"{agent_name}-A{attempt}",
                    request_timeout=self.runtime.request_timeout,
                    tool_choice={"type": "function", "function": {"name": expected_name}},
                )
            except Exception as exc:
                self.model_call_count += 1
                last_error = f"model call failed: {type(exc).__name__}: {str(exc)[:500]}"
                last_normalized = {"valid": False, "errors": [last_error]}
                self.runtime.log_agent(
                    round_log,
                    agent_name if attempt == 1 else f"{agent_name}Retry{attempt}",
                    role,
                    messages,
                    last_error,
                    0,
                    0,
                )
                self.runtime.temp_log_event(
                    "quality_agent_call_failed",
                    round_log,
                    agent=agent_name,
                    role=role,
                    attempt=attempt,
                    expected_tool=expected_name,
                    error=last_error,
                )
                # call_model already exhausted the configured provider pool and
                # per-provider retries. Replaying the identical structured task
                # only multiplies a pool-wide protocol failure.
                break
            self.model_call_count += 1
            total_pec += pec
            total_ec += ec
            parsed_args = {}
            parse_note = ""
            for tool_call in tool_calls or []:
                function = tool_call.get("function", {}) or {}
                if function.get("name") != expected_name:
                    continue
                parsed_args, parse_note = self.runtime.parse_tool_arguments(
                    expected_name,
                    function.get("arguments", "{}"),
                )
                if parsed_args:
                    break
            valid = False
            normalized = None
            if parsed_args and not parse_note:
                valid, normalized = validator(parsed_args)
            elif parsed_args:
                valid, normalized = validator(parsed_args)
            if not valid:
                validation_errors = (normalized or {}).get("errors", []) if isinstance(normalized, dict) else []
                last_error = parse_note or "; ".join(validation_errors[:8]) or "invalid structured result"
            last_args = parsed_args or last_args
            last_normalized = normalized or last_normalized
            self.runtime.log_agent(
                round_log,
                agent_name if attempt == 1 else f"{agent_name}Retry{attempt}",
                role,
                messages,
                text,
                pec,
                ec,
                tool_calls_list=(
                    [{"function": expected_name, "arguments": parsed_args}]
                    if parsed_args else None
                ),
            )
            if valid:
                return parsed_args, normalized, total_pec, total_ec
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your structured result was rejected: {last_error}. Re-read the complete packet and call "
                        f"{expected_name} again with every required field."
                    ),
                }
            )
        return last_args, last_normalized, total_pec, total_ec

    def _run_investigator(
        self,
        agent_name: str,
        instruction: str,
        target_packet_text: str,
        neutral_cases: list[dict[str, Any]],
        neutral_sequences: list[dict[str, Any]],
        packet: dict[str, Any],
        round_log: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int]:
        cases_text = self._format_neutral_cases(neutral_cases)
        system_prompt = (
            instruction
            + "\nUse only raw novel evidence. Candidate cases are unordered hypotheses; their order and labels carry no "
            "support count, and all may be wrong. You may propose a Chinese source speaker absent from the cases. "
            "Do not use outside knowledge, character personality, occupation, species, topic, or presumed speech style. "
            "Do not force alternation. An action immediately before a quote may cause the other participant's response; "
            "call it explicit_attribution only when a speech verb grammatically binds the proposed speaker. "
            f"If nobody utters the exact target span because it is unspoken thought, narration, displayed text, or an "
            f"object sound, set voice_kind=non_person and speaker={self.runtime.non_person_label}, even when a person's "
            "mind or action is the source. A named person's laugh, sob, groan, cough, snore, or other bodily/vocal "
            "sound is person voice and keeps that person's name. Speech interrupted after it starts is also person "
            "voice; distinguish it from a sentence that was never spoken. For this annotation schema, quoted meaning "
            "explicitly conveyed by a named person's gaze, face, gesture, or expression with wording such as 'as if "
            "saying' is attributed to that person as person/embedded_quote, even when it is not audible speech. "
            "Speech by unnamed human passersby, workers, customers, or a crowd is still person voice; use a stable "
            "source role such as 路人 or 人群 rather than non_person. Scan later narration for delayed attribution of "
            "a whole block of consecutive quotes. "
            "Use read_novel_lines to expand context and search_novel only for identity or "
            "alias evidence. When evidence is sufficient call submit_target_evidence. Return unresolved rather than guess."
        )
        sequences_text = self._format_neutral_sequences(neutral_sequences)
        user_content = target_packet_text + "\n\n" + cases_text + "\n\n" + sequences_text
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = [self.runtime.read_tool, self.runtime.search_tool, TARGET_EVIDENCE_TOOL]
        available_lines = {item["line"] for item in packet["raw_lines"]}
        total_pec = 0
        total_ec = 0
        tool_log = []
        final_args: dict[str, Any] = {}
        final_card = self._unresolved_card("investigator did not submit valid evidence")
        force_submission = False

        for round_index in range(max(1, self.runtime.deep_tool_rounds)):
            messages = self._compact_messages(messages)
            active_tools = [TARGET_EVIDENCE_TOOL] if force_submission else tools
            tool_choice: Any = (
                {"type": "function", "function": {"name": "submit_target_evidence"}}
                if force_submission
                else "auto"
            )
            try:
                text, pec, ec, tool_calls = self.runtime.call_model(
                    messages,
                    tools=active_tools,
                    label=f"{agent_name}-R{round_index + 1}",
                    request_timeout=max(self.runtime.request_timeout, 180),
                    tool_choice=tool_choice,
                )
            except Exception as exc:
                self.model_call_count += 1
                error_text = f"model call failed: {type(exc).__name__}: {str(exc)[:500]}"
                self.runtime.log_agent(
                    round_log,
                    f"{agent_name}Failure{round_index + 1}",
                    "target_evidence_investigator_failure",
                    messages,
                    error_text,
                    0,
                    0,
                )
                self.runtime.temp_log_event(
                    "quality_investigator_call_failed",
                    round_log,
                    agent=agent_name,
                    tool_round=round_index + 1,
                    error=error_text,
                )
                if force_submission:
                    final_card["reason"] = f"forced submission failed after provider-pool exhaustion: {error_text}"
                    return final_card, total_pec, total_ec
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous model call failed: {error_text}. Continue the investigation with tools, "
                            "or submit unresolved when evidence cannot be established."
                        ),
                    }
                )
                continue
            self.model_call_count += 1
            total_pec += pec
            total_ec += ec
            parsed_calls = []
            submitted = None
            for tool_call in tool_calls or []:
                function = tool_call.get("function", {}) or {}
                name = function.get("name", "")
                args, parse_error = self.runtime.parse_tool_arguments(
                    name,
                    function.get("arguments", "{}"),
                )
                parsed_calls.append((tool_call, name, args, parse_error))
                if name == "submit_target_evidence" and args:
                    submitted = args

            if submitted:
                card = self._normalize_target_card(submitted, available_lines)
                final_args = submitted
                if card["valid"]:
                    final_card = card
                    self.runtime.log_agent(
                        round_log,
                        agent_name,
                        "target_evidence_investigator",
                        messages,
                        text,
                        pec,
                        ec,
                        total_pec=total_pec,
                        total_ec=total_ec,
                        tool_calls_list=tool_log + [{"function": "submit_target_evidence", "arguments": submitted}],
                    )
                    return final_card, total_pec, total_ec

            if parsed_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            self.runtime.normalize_tool_call(tool_call, args)
                            for tool_call, _, args, _ in parsed_calls
                        ],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You did not use a tool. Stop explaining and call submit_target_evidence now; "
                            "use unresolved if the supplied evidence is insufficient."
                        ),
                    }
                )
                force_submission = True
                continue

            force_submission = False

            for _, name, args, parse_error in parsed_calls:
                if name == "read_novel_lines":
                    try:
                        start = max(1, int(args.get("start", 1)))
                        count = max(1, min(80, int(args.get("count", 20))))
                    except (TypeError, ValueError):
                        start, count = 1, 20
                    result = self.runtime.read_novel_lines(start, count)
                    result = result[:16000]
                    available_lines.update(range(start, start + count))
                    tool_log.append({"function": name, "arguments": {"start": start, "count": count}})
                    self.runtime.trace_tool_result(
                        agent_name,
                        round_index + 1,
                        name,
                        {"start": start, "count": count},
                        result,
                    )
                    messages.append({"role": "tool", "content": result})
                elif name == "search_novel":
                    keyword = str(args.get("keyword") or "").strip()
                    try:
                        context_lines = max(0, min(5, int(args.get("context_lines", 2))))
                    except (TypeError, ValueError):
                        context_lines = 2
                    result = self.runtime.search_novel(keyword, context_lines) if keyword else "keyword is required"
                    result = result[:16000]
                    available_lines.update(
                        int(value)
                        for value in re.findall(r"(?:>>>|\s+)\s*L?(\d+):", result)
                    )
                    tool_log.append(
                        {"function": name, "arguments": {"keyword": keyword, "context_lines": context_lines}}
                    )
                    self.runtime.trace_tool_result(
                        agent_name,
                        round_index + 1,
                        name,
                        {"keyword": keyword, "context_lines": context_lines},
                        result,
                    )
                    messages.append({"role": "tool", "content": result})
                elif name != "submit_target_evidence":
                    messages.append({"role": "tool", "content": f"Unsupported tool {name}: {parse_error}"})

            if total_pec + total_ec >= self.runtime.deep_token_budget:
                break

        final_messages = self._compact_messages(messages)
        final_messages.append(
            {
                "role": "user",
                "content": "Stop expanding context. Call submit_target_evidence now; use unresolved if evidence is insufficient.",
            }
        )
        try:
            text, pec, ec, tool_calls = self.runtime.call_model(
                final_messages,
                tools=[TARGET_EVIDENCE_TOOL],
                label=f"{agent_name}-Final",
                request_timeout=max(self.runtime.request_timeout, 180),
                tool_choice={"type": "function", "function": {"name": "submit_target_evidence"}},
            )
        except Exception as exc:
            self.model_call_count += 1
            error_text = f"final model call failed: {type(exc).__name__}: {str(exc)[:500]}"
            self.runtime.log_agent(
                round_log,
                f"{agent_name}FinalFailure",
                "target_evidence_investigator_failure",
                final_messages,
                error_text,
                0,
                0,
                total_pec=total_pec,
                total_ec=total_ec,
                tool_calls_list=tool_log or None,
            )
            self.runtime.temp_log_event(
                "quality_investigator_final_failed",
                round_log,
                agent=agent_name,
                error=error_text,
            )
            final_card["reason"] = error_text
            return final_card, total_pec, total_ec
        self.model_call_count += 1
        total_pec += pec
        total_ec += ec
        for tool_call in tool_calls or []:
            function = tool_call.get("function", {}) or {}
            if function.get("name") != "submit_target_evidence":
                continue
            args, _ = self.runtime.parse_tool_arguments(
                "submit_target_evidence",
                function.get("arguments", "{}"),
            )
            card = self._normalize_target_card(args, available_lines)
            if card["valid"]:
                final_args = args
                final_card = card
                break
        self.runtime.log_agent(
            round_log,
            agent_name,
            "target_evidence_investigator",
            final_messages,
            text,
            pec,
            ec,
            total_pec=total_pec,
            total_ec=total_ec,
            tool_calls_list=tool_log + (
                [{"function": "submit_target_evidence", "arguments": final_args}]
                if final_args else []
            ),
        )
        return final_card, total_pec, total_ec

    def _run_verdict(
        self,
        target_packet_text: str,
        neutral_cases: list[dict[str, Any]],
        neutral_sequences: list[dict[str, Any]],
        investigations: list[dict[str, Any]],
        packet: dict[str, Any],
        round_log: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int]:
        evidence_blocks = [
            self._format_neutral_cases(neutral_cases),
            self._format_neutral_sequences(neutral_sequences),
            "[Independent evidence investigations]",
        ]
        for index, report in enumerate(investigations, 1):
            evidence_blocks.append(
                f"  Investigation {index}: status={report.get('status')}; speaker={report.get('speaker') or '?'}; "
                f"voice={report.get('voice_kind')}; quote_type={report.get('quote_type')}; "
                f"basis={report.get('evidence_basis')}; confidence={report.get('confidence')}; "
                f"citations={report.get('citations')}; reason={report.get('reason')}"
            )
        user_content = target_packet_text + "\n\n" + "\n".join(evidence_blocks)
        system_prompt = (
            "Act as an evidence judge, not a voter. Candidate cases are deliberately unordered and expose no source "
            "or support counts. Re-read the exact raw scene and test every case for direct contradiction. A majority, "
            "topic, persona, occupation, species, characteristic vocabulary, or assumed alternation is not evidence. "
            "An adjacent action without a speech verb is not direct attribution and may instead prompt the other "
            "participant's reply. Label direct only when a cited speech verb grammatically binds the proposed speaker. "
            f"Judge the exact span's voice_kind separately from whose mind or action contains it. For unspoken thought, "
            f"narration, displayed text, or object sound, use voice_kind=non_person and speaker={self.runtime.non_person_label}. "
            "A named person's laugh, sob, groan, cough, snore, or bodily/vocal sound remains person voice. Words already "
            "started before an interruption also remain person voice, unlike words explicitly never spoken. For this "
            "annotation schema, quoted meaning explicitly conveyed by a named person's gaze, face, gesture, or "
            "expression with wording such as 'as if saying' belongs to that person as person/embedded_quote. "
            "Unnamed human passersby or crowds remain person voice; later narration may attribute a whole preceding "
            "block of quotes to them. Use a stable human role such as 路人 or 人群, not non_person. "
            "You may select a new Chinese source speaker absent from the cases. Resolve only when citations bind the "
            "exact target quote or a complete scene sequence plus independent investigations excludes alternatives. "
            "Otherwise return unresolved. Call submit_scene_verdict immediately without prose."
        )

        def validator(args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
            card = self._normalize_verdict(args, {item["line"] for item in packet["raw_lines"]})
            return card["valid"], card

        _, card, pec, ec = self._run_required_tool(
            "SceneEvidenceJudge",
            "source_blind_evidence_judge",
            system_prompt,
            user_content,
            SCENE_VERDICT_TOOL,
            "submit_scene_verdict",
            validator,
            round_log,
        )
        return card or self._unresolved_verdict("invalid final verdict"), pec, ec

    def _run_contrastive_boundary(
        self,
        baseline: str,
        scene_cards: list[dict[str, Any]],
        investigations: list[dict[str, Any]],
        verdict: dict[str, Any],
        packet: dict[str, Any],
        dialogue_index: int,
        target_packet_text: str,
        round_log: dict[str, Any],
        neighbor_candidates: list[str] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        candidates = self._contrastive_candidate_pair(
            baseline,
            scene_cards,
            investigations,
            verdict,
            neighbor_candidates,
        )
        if len(candidates) != 2:
            return {
                "valid": False,
                "status": "not_run",
                "speaker": "",
                "unanimous": False,
                "candidates": candidates,
                "briefs": [],
                "juries": [],
                "reason": "fewer than two distinct person candidates",
            }, 0, 0
        generic_flags = [self.runtime.is_generic_speaker(item["speaker"]) for item in candidates]
        if generic_flags[0] != generic_flags[1]:
            return {
                "valid": False,
                "status": "not_run",
                "speaker": "",
                "unanimous": False,
                "candidates": candidates,
                "briefs": [],
                "juries": [],
                "referees": [],
                "consensus_deciders": [],
                "consensus_source": "none",
                "reason": "generic-role versus named-identity conflict belongs to identity canonicalization",
            }, 0, 0

        available_lines = {item["line"] for item in packet["raw_lines"]}
        total_pec = 0
        total_ec = 0
        briefs = []
        for index, candidate in enumerate(candidates):
            hypothesis_id = "A" if index == 0 else "B"
            system_prompt = (
                "Act as a skeptical advocate for exactly the assigned target-speaker hypothesis. Build the strongest "
                "source-grounded case, but also state its weakest unsupported assumption and the strongest raw-text "
                "contradiction. Reconstruct the floor before and after the target. Compare whether the target is a "
                "continuation, answer, correction, reformulation, or setup for a neighboring quote. Narration about a "
                "listener's reaction may close the prior speaker's turn; an attribution attached to a later quote is "
                "only an anchor for that later quote. A non-colon speech attribution immediately after the previous "
                "quote is consumed by that previous quote and cannot be moved forward onto TARGET. A paraphrase or "
                "'in other words' reformulation may be same-speaker elaboration; call it a response by another person "
                "only when address, answer structure, or narration proves a switch. Do not use personality, occupation, species, remembered plot, "
                "characteristic vocabulary, or mechanical alternation. Call submit_boundary_brief without prose."
            )
            user_content = (
                target_packet_text
                + "\n\n[Assigned hypothesis]\n"
                + f"Hypothesis-{hypothesis_id}: the exact TARGET span is spoken by {candidate['speaker']}.\n"
                + "Defend only this hypothesis, while reporting its real weakness honestly."
            )

            def validator(
                args: dict[str, Any],
                expected_id: str = hypothesis_id,
            ) -> tuple[bool, dict[str, Any]]:
                card = self._normalize_boundary_brief(args, expected_id, available_lines)
                return card["valid"], card

            _, brief, pec, ec = self._run_required_tool(
                f"BoundaryHypothesis{hypothesis_id}Advocate",
                "contrastive_boundary_advocate",
                system_prompt,
                user_content,
                BOUNDARY_BRIEF_TOOL,
                "submit_boundary_brief",
                validator,
                round_log,
            )
            total_pec += pec
            total_ec += ec
            normalized = brief or self._invalid_boundary_brief(hypothesis_id)
            normalized = dict(normalized)
            normalized["speaker"] = candidate["speaker"]
            briefs.append(normalized)

        jury_specs = [
            (
                "BoundaryForwardJury",
                (0, 1),
                "Read forward. Locate the last reliable speaker anchor before TARGET and the first reliable anchor "
                "after it, then count only transitions justified by raw narration or dialogue acts.",
            ),
            (
                "BoundaryReverseJury",
                (1, 0),
                "Read backward from the first reliable anchor after TARGET. Test which hypothesis reconstructs the "
                "preceding floor with fewer unsupported switches. The A/B order is intentionally reversed.",
            ),
            (
                "BoundaryRhetoricalJury",
                (0, 1),
                "Compare TARGET with both neighboring utterances for answer, correction, paraphrase, elaboration, "
                "shared proposition, and listener-reaction boundaries. Prefer a complete rhetorical chain over mere "
                "physical proximity to the previous named speaker.",
            ),
        ]
        juries = []
        for agent_name, order, instruction in jury_specs:
            local_candidates = [candidates[order[0]], candidates[order[1]]]
            local_briefs = [briefs[order[0]], briefs[order[1]]]
            hypotheses = []
            for local_index, (candidate, brief) in enumerate(zip(local_candidates, local_briefs)):
                local_id = "A" if local_index == 0 else "B"
                hypotheses.append(
                    f"Hypothesis-{local_id}: TARGET speaker={candidate['speaker']}\n"
                    f"  advocate_support={brief.get('supporting_citations', [])}: "
                    f"{brief.get('strongest_case', '')}\n"
                    f"  advocate_contradiction={brief.get('contradicting_citations', [])}; "
                    f"weakest_assumption={brief.get('weakest_assumption', '')}"
                )
            system_prompt = (
                "Act as a contrastive discourse-boundary judge. There are exactly two mutually exclusive hypotheses. "
                "Do not vote, average earlier agents, or infer support from ordering. Independently verify every brief "
                "against the raw scene. A person may speak consecutive turns. A listener action is not attribution. "
                "Explicit quote binding wins; otherwise reconstruct semantic and rhetorical continuity across the full "
                "local block. A non-colon attribution immediately after the previous quote belongs to that previous "
                "quote and is not evidence for TARGET. A later paraphrase, summary, or 'in other words' formulation "
                "does not prove a speaker switch; test same-speaker elaboration against other-speaker response and "
                "require textual evidence for the transition. Choose unresolved if neither hypothesis excludes the other. "
                + instruction
                + " Call submit_boundary_verdict without prose."
            )
            user_content = target_packet_text + "\n\n[Competing hypotheses]\n" + "\n".join(hypotheses)

            def validator(
                args: dict[str, Any],
                mapped_candidates: list[dict[str, Any]] = local_candidates,
            ) -> tuple[bool, dict[str, Any]]:
                card = self._normalize_boundary_jury(args, mapped_candidates, available_lines)
                return card["valid"], card

            _, jury, pec, ec = self._run_required_tool(
                agent_name,
                "contrastive_boundary_jury",
                system_prompt,
                user_content,
                BOUNDARY_JURY_TOOL,
                "submit_boundary_verdict",
                validator,
                round_log,
            )
            total_pec += pec
            total_ec += ec
            juries.append(jury or self._invalid_boundary_jury("invalid structured jury result"))

        resolved = [
            jury for jury in juries
            if jury.get("valid") and jury.get("status") == "resolved" and jury.get("speaker")
        ]
        initial_unanimous = bool(
            len(resolved) == len(jury_specs)
            and all(
                self.runtime.same_speaker(jury["speaker"], resolved[0]["speaker"])
                for jury in resolved[1:]
            )
        )
        referees = []
        referee_unanimous = False
        referee_resolved: list[dict[str, Any]] = []
        if not initial_unanimous:
            prior_analyses = "\n".join(
                f"Analysis {index}: previous_relation={jury.get('target_to_previous')}; "
                f"next_relation={jury.get('target_to_next')}; confidence={jury.get('confidence')}; "
                f"reason={jury.get('reason')}"
                for index, jury in enumerate(juries, 1)
            )
            referee_specs = [
                (
                    "BoundaryAssumptionReferee",
                    (0, 1),
                    "Identify the single weakest unsupported transition assumption in each hypothesis. Select only if "
                    "one reconstruction requires strictly fewer unsupported speaker changes.",
                ),
                (
                    "BoundaryCounterfactualReferee",
                    (1, 0),
                    "Reconstruct the entire local dialogue twice, once under each target assignment, reading backward "
                    "from the next anchored quote. The A/B presentation order is reversed.",
                ),
            ]
            for agent_name, order, instruction in referee_specs:
                local_candidates = [candidates[order[0]], candidates[order[1]]]
                local_briefs = [briefs[order[0]], briefs[order[1]]]
                hypotheses = []
                for local_index, (candidate, brief) in enumerate(zip(local_candidates, local_briefs)):
                    local_id = "A" if local_index == 0 else "B"
                    hypotheses.append(
                        f"Hypothesis-{local_id}: TARGET speaker={candidate['speaker']}\n"
                        f"  strongest_case={brief.get('strongest_case', '')}\n"
                        f"  weakest_assumption={brief.get('weakest_assumption', '')}"
                    )
                system_prompt = (
                    "Act as a disagreement referee, not a voter. Earlier analyses are fallible arguments; their count, "
                    "confidence labels, order, and recommendations carry no authority. Check every premise against raw "
                    "text and the deterministic boundary constraints. A post-quote non-colon attribution is already "
                    "consumed by the preceding quote. Missing narration does not prove continuation. A semantic "
                    "paraphrase can be same-speaker elaboration or another-speaker response; compare both and require "
                    "textual evidence for any switch. Do not use characteristic vocabulary or role stereotypes. "
                    + instruction
                    + " Call submit_boundary_verdict without prose."
                )
                user_content = (
                    target_packet_text
                    + "\n\n[Competing hypotheses]\n"
                    + "\n".join(hypotheses)
                    + "\n\n[Fallible prior analyses - critique, do not vote]\n"
                    + prior_analyses
                )

                def validator(
                    args: dict[str, Any],
                    mapped_candidates: list[dict[str, Any]] = local_candidates,
                ) -> tuple[bool, dict[str, Any]]:
                    card = self._normalize_boundary_jury(args, mapped_candidates, available_lines)
                    return card["valid"], card

                _, referee, pec, ec = self._run_required_tool(
                    agent_name,
                    "contrastive_boundary_referee",
                    system_prompt,
                    user_content,
                    BOUNDARY_JURY_TOOL,
                    "submit_boundary_verdict",
                    validator,
                    round_log,
                )
                total_pec += pec
                total_ec += ec
                referees.append(referee or self._invalid_boundary_jury("invalid structured referee result"))
            referee_resolved = [
                referee for referee in referees
                if referee.get("valid")
                and referee.get("status") == "resolved"
                and referee.get("speaker")
            ]
            referee_unanimous = bool(
                len(referee_resolved) == len(referee_specs)
                and self.runtime.same_speaker(
                    referee_resolved[0]["speaker"],
                    referee_resolved[1]["speaker"],
                )
            )

        consensus_deciders = resolved if initial_unanimous else referee_resolved if referee_unanimous else []
        unanimous = bool(consensus_deciders)
        winner = consensus_deciders[0]["speaker"] if unanimous else ""
        citations = sorted(
            {citation for jury in consensus_deciders for citation in jury.get("citations", [])}
        )
        confidence = (
            "high" if unanimous and sum(
                jury.get("confidence") == "high" for jury in consensus_deciders
            ) >= 2
            else "medium" if unanimous else "low"
        )
        return {
            "valid": unanimous,
            "status": "resolved" if unanimous else "unresolved",
            "speaker": winner,
            "unanimous": unanimous,
            "confidence": confidence,
            "citations": citations,
            "candidates": candidates,
            "briefs": briefs,
            "juries": juries,
            "referees": referees,
            "consensus_deciders": consensus_deciders,
            "consensus_source": (
                "initial_juries" if initial_unanimous
                else "disagreement_referees" if referee_unanimous
                else "none"
            ),
            "reason": (
                "all mirrored boundary juries selected the same source speaker"
                if initial_unanimous
                else "both disagreement referees selected the same source speaker"
                if referee_unanimous
                else "boundary juries and disagreement referees were incomplete or disagreed"
            ),
        }, total_pec, total_ec

    def _format_boundary_constraints(
        self,
        packet: dict[str, Any],
        dialogue_index: int,
    ) -> str:
        turns = packet.get("turns") or []
        target_offset = next(
            (index for index, turn in enumerate(turns) if turn.get("dialogue_index") == dialogue_index),
            -1,
        )
        if target_offset < 0:
            return "[Deterministic quote-boundary constraints]\n  Target turn was not found."
        target = turns[target_offset]
        previous = turns[target_offset - 1] if target_offset > 0 else None
        following = turns[target_offset + 1] if target_offset + 1 < len(turns) else None
        raw_lines = packet.get("raw_lines") or []
        raw_between_previous = [
            item for item in raw_lines
            if previous and previous["line"] < item["line"] < target["line"]
        ]
        raw_between_following = [
            item for item in raw_lines
            if following and target["line"] < item["line"] < following["line"]
        ]

        def turn_label(turn: dict[str, Any] | None) -> str:
            if not turn:
                return "none"
            turn_id = str(turn.get("turn_id") or f"dialogue#{turn.get('dialogue_index', '?')}")
            return f"{turn_id} L{turn.get('line', '?')}"

        notes = [
            "[Deterministic quote-boundary constraints]",
            f"  TARGET={turn_label(target)}; previous={turn_label(previous)}; next={turn_label(following)}.",
        ]

        def has_speech_binding(text: str) -> bool:
            return bool(re.search(_SPEECH_VERB_PATTERN, text))

        if raw_between_previous:
            first = raw_between_previous[0]
            text = str(first.get("text") or "").strip()
            if has_speech_binding(text):
                if re.search(r"[:：]\s*$", text):
                    notes.append(
                        f"  L{first['line']} ends with a colon and may introduce a following quote; verify its exact scope."
                    )
                else:
                    notes.append(
                        f"  L{first['line']} is the first narration after {previous.get('turn_id')} and contains a "
                        f"non-colon speech attribution. It binds that preceding quote, not TARGET."
                    )
            last = raw_between_previous[-1]
            last_text = str(last.get("text") or "").strip()
            if has_speech_binding(last_text) and re.search(r"[:：]\s*$", last_text):
                notes.append(
                    f"  L{last['line']} immediately precedes TARGET with a colon and may introduce TARGET."
                )
        if raw_between_following:
            first = raw_between_following[0]
            text = str(first.get("text") or "").strip()
            if has_speech_binding(text) and not re.search(r"[:：]\s*$", text):
                notes.append(
                    f"  L{first['line']} is the first narration after TARGET and contains a non-colon speech "
                    f"attribution. It may bind TARGET, not the next quote."
                )
            last = raw_between_following[-1]
            last_text = str(last.get("text") or "").strip()
            if has_speech_binding(last_text) and re.search(r"[:：]\s*$", last_text):
                notes.append(
                    f"  L{last['line']} immediately precedes {following.get('turn_id')} with a colon. It introduces "
                    f"that next quote and does not directly bind TARGET."
                )
        if len(notes) == 2:
            notes.append("  No deterministic adjacent speech-attribution scope was found; use discourse evidence conservatively.")
        notes.append(
            "  Listener reactions and semantic paraphrases may mark a boundary, but neither alone proves a speaker switch."
        )
        return "\n".join(notes)

    @staticmethod
    def _contact_verbs_for_speaker(text: str, speaker: str) -> list[str]:
        if not text or not speaker:
            return []
        escaped = re.escape(speaker)
        contact_pattern = "|".join(sorted(_DIALOGUE_CONTACT_VERBS, key=len, reverse=True))
        found = []
        subject_patterns = (
            rf"(?<![向对朝跟]){escaped}[^。！？；]{{0,80}}?(?P<verb>{contact_pattern})(?!的是)",
            rf"(?P<verb>{contact_pattern})(?:的)?是[^。！？；]{{0,12}}?{escaped}(?:[，。！？；]|$)",
        )
        for pattern in subject_patterns:
            for match in re.finditer(pattern, text):
                verb = match.group("verb")
                if verb not in found:
                    found.append(verb)
        return found

    def _baseline_scope_conflict(
        self,
        baseline: str,
        baseline_evidence: dict[str, Any],
        baseline_reason: str,
        packet: dict[str, Any],
        dialogue_index: int,
        local_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        turns = packet.get("turns") or []
        target_offset = next(
            (index for index, turn in enumerate(turns) if turn.get("dialogue_index") == dialogue_index),
            -1,
        )
        audit = {
            "conflict": False,
            "consumed_lines": [],
            "consumed_speakers": [],
            "consumed_verbs": [],
            "reason": "no consumed previous-quote attribution conflicts with the baseline evidence",
        }
        if target_offset <= 0:
            return audit

        previous = turns[target_offset - 1]
        target = turns[target_offset]
        between = [
            item for item in packet.get("raw_lines") or []
            if previous["line"] < item["line"] < target["line"]
        ]
        if not between:
            return audit
        first = between[0]
        first_text = str(first.get("text") or "").strip()
        if not re.search(_SPEECH_VERB_PATTERN, first_text) or re.search(r"[:：]\s*$", first_text):
            return audit

        consumed_line = int(first["line"])
        baseline_clean = self.runtime.validate_speaker(baseline) or baseline
        consumed_speakers = []
        matched_baseline_verbs = []
        for item in (local_evidence or {}).get("attribution_candidates") or []:
            try:
                item_line = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            if item_line != consumed_line:
                continue
            speaker = self.runtime.validate_speaker(item.get("speaker", "")) or ""
            if speaker and not any(self.runtime.same_speaker(speaker, value) for value in consumed_speakers):
                consumed_speakers.append(speaker)
            verb = str(item.get("verb") or "").strip()
            if (
                speaker
                and self.runtime.same_speaker(baseline_clean, speaker)
                and verb
                and verb not in matched_baseline_verbs
            ):
                matched_baseline_verbs.append(verb)

        contact_verbs = self._contact_verbs_for_speaker(first_text, baseline_clean)
        if contact_verbs and not any(
            self.runtime.same_speaker(baseline_clean, speaker) for speaker in consumed_speakers
        ):
            consumed_speakers.append(baseline_clean)
        for verb in contact_verbs:
            if verb not in matched_baseline_verbs:
                matched_baseline_verbs.append(verb)

        audit["consumed_lines"] = [consumed_line]
        audit["consumed_speakers"] = consumed_speakers
        audit["consumed_verbs"] = matched_baseline_verbs
        basis = str(baseline_evidence.get("evidence_basis") or "unknown")
        reason_cites_line = bool(re.search(rf"(?<!\d)L?{consumed_line}(?!\d)", baseline_reason or ""))
        baseline_matches_consumed = any(
            self.runtime.same_speaker(baseline_clean, speaker) for speaker in consumed_speakers
        )
        if (
            baseline_matches_consumed
            and reason_cites_line
            and basis in {"explicit_attribution", "speaker_action", "sequence_constraint"}
        ):
            audit["conflict"] = True
            audit["reason"] = (
                f"baseline cites L{consumed_line} for {basis}, but that non-colon attribution is consumed by "
                "the preceding quote"
            )
        return audit

    def _scope_repair_consensus(
        self,
        baseline: str,
        scene_cards: list[dict[str, Any]],
        investigations: list[dict[str, Any]],
        target_line: int,
    ) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        baseline_clean = self.runtime.validate_speaker(baseline) or baseline

        def add(card: dict[str, Any], source_kind: str, source_name: str) -> None:
            if not card.get("valid"):
                return
            if source_kind == "investigation" and card.get("status") != "resolved":
                return
            if card.get("confidence") not in {"high", "medium"}:
                return
            speaker = self.runtime.validate_speaker(card.get("speaker", "")) or ""
            if (
                not speaker
                or not self.runtime.is_valid_speaker(speaker)
                or self.runtime.same_speaker(speaker, baseline_clean)
            ):
                return
            citations = self._int_list(card.get("citations"), None)
            if not citations or not any(abs(line - target_line) <= 8 for line in citations):
                return
            group = next(
                (item for item in groups if self.runtime.same_speaker(item["speaker"], speaker)),
                None,
            )
            if group is None:
                group = {
                    "speaker": speaker,
                    "sources": [],
                    "source_kinds": set(),
                    "citations": set(),
                }
                groups.append(group)
            if source_name not in group["sources"]:
                group["sources"].append(source_name)
            group["source_kinds"].add(source_kind)
            group["citations"].update(citations)

        for card in scene_cards:
            add(card, "scene", str(card.get("agent") or "scene-mapper"))
        for card in investigations:
            add(card, "investigation", str(card.get("agent") or "investigator"))

        qualified = [
            group for group in groups
            if len(group["sources"]) >= 3
            and (
                {"scene", "investigation"}.issubset(group["source_kinds"])
                or sum(
                    str(source).startswith("TargetContactScopeRepair")
                    for source in group["sources"]
                ) >= 2
            )
        ]
        qualified.sort(key=lambda item: len(item["sources"]), reverse=True)
        if not qualified:
            return {"valid": False, "speaker": "", "support": 0, "sources": [], "citations": []}
        winner = qualified[0]
        if len(qualified) > 1 and len(qualified[0]["sources"]) == len(qualified[1]["sources"]):
            return {"valid": False, "speaker": "", "support": 0, "sources": [], "citations": []}
        return {
            "valid": True,
            "speaker": winner["speaker"],
            "support": len(winner["sources"]),
            "sources": winner["sources"],
            "citations": sorted(winner["citations"]),
            "scene_support": sum(source in {card.get("agent") for card in scene_cards} for source in winner["sources"]),
            "investigation_support": sum(
                source in {card.get("agent") for card in investigations} for source in winner["sources"]
            ),
            "contact_repair_support": sum(
                str(source).startswith("TargetContactScopeRepair") for source in winner["sources"]
            ),
        }

    def _contrastive_candidate_pair(
        self,
        baseline: str,
        scene_cards: list[dict[str, Any]],
        investigations: list[dict[str, Any]],
        verdict: dict[str, Any],
        neighbor_candidates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []

        def add(value: str, source: str, weight: int) -> None:
            speaker = self.runtime.validate_speaker(value) or ""
            if (
                not speaker
                or not self.runtime.is_valid_speaker(speaker)
                or speaker == self.runtime.non_person_label
            ):
                return
            group = next(
                (item for item in groups if self.runtime.same_speaker(item["speaker"], speaker)),
                None,
            )
            if group is None:
                groups.append({"speaker": speaker, "score": weight, "sources": [source]})
            else:
                group["score"] += weight
                if source not in group["sources"]:
                    group["sources"].append(source)

        add(baseline, "baseline", 2)
        if verdict.get("status") == "resolved":
            add(verdict.get("speaker", ""), "preliminary_verdict", 3)
        for card in scene_cards:
            if card.get("valid"):
                add(card.get("speaker", ""), f"scene:{card.get('agent') or 'mapper'}", 2)
        for report in investigations:
            if report.get("valid") and report.get("status") == "resolved":
                add(report.get("speaker", ""), f"investigation:{report.get('agent') or 'agent'}", 2)
        for speaker in neighbor_candidates or []:
            add(speaker, "neighboring-scene-turn", 1)

        if len(groups) < 2:
            return groups
        groups.sort(key=lambda item: (item["score"], len(item["sources"])), reverse=True)
        baseline_clean = self.runtime.validate_speaker(baseline) or ""
        baseline_group = next(
            (item for item in groups if self.runtime.same_speaker(item["speaker"], baseline_clean)),
            None,
        )
        if baseline_group:
            alternative = next(
                item for item in groups
                if not self.runtime.same_speaker(item["speaker"], baseline_group["speaker"])
            )
            return [baseline_group, alternative]
        return groups[:2]

    def _neighboring_scene_candidates(
        self,
        scene_solution: dict[str, Any],
        dialogue_index: int,
        baseline: str,
    ) -> list[str]:
        packet = scene_solution.get("packet") or {}
        turn_by_id = {
            str(turn.get("turn_id") or turn.get("id")): turn for turn in packet.get("turns") or []
        }
        groups: list[dict[str, Any]] = []
        baseline_clean = self.runtime.validate_speaker(baseline) or baseline

        for scene_map in scene_solution.get("maps") or []:
            if not scene_map.get("valid"):
                continue
            for turn_id, assignment in (scene_map.get("assignments") or {}).items():
                turn = turn_by_id.get(str(turn_id))
                if not turn:
                    continue
                distance = abs(int(turn.get("dialogue_index", -999999)) - int(dialogue_index))
                if distance == 0 or distance > 3:
                    continue
                speaker = self.runtime.validate_speaker(assignment.get("speaker", "")) or ""
                if (
                    not speaker
                    or not self.runtime.is_valid_speaker(speaker)
                    or speaker == self.runtime.non_person_label
                    or self.runtime.same_speaker(speaker, baseline_clean)
                ):
                    continue
                group = next(
                    (item for item in groups if self.runtime.same_speaker(item["speaker"], speaker)),
                    None,
                )
                if group is None:
                    group = {"speaker": speaker, "score": 0, "sources": set()}
                    groups.append(group)
                group["score"] += 4 - distance
                group["sources"].add(str(scene_map.get("agent") or "scene-mapper"))

        groups.sort(
            key=lambda item: (item["score"], len(item["sources"])),
            reverse=True,
        )
        return [item["speaker"] for item in groups[:3]]

    @staticmethod
    def _normalize_boundary_brief(
        args: dict[str, Any],
        expected_id: str,
        available_lines: set[int],
    ) -> dict[str, Any]:
        supporting = SceneSequenceAudit._int_list(args.get("supporting_citations"), available_lines)
        contradicting = SceneSequenceAudit._int_list(args.get("contradicting_citations"), available_lines)
        strongest_case = str(args.get("strongest_case") or "").strip()
        weakest_assumption = str(args.get("weakest_assumption") or "").strip()
        confidence = str(args.get("confidence") or "low")
        valid = bool(
            args.get("hypothesis_id") == expected_id
            and strongest_case
            and weakest_assumption
            and confidence in {"high", "medium", "low"}
        )
        return {
            "valid": valid,
            "hypothesis_id": expected_id,
            "supporting_citations": supporting,
            "contradicting_citations": contradicting,
            "strongest_case": strongest_case,
            "weakest_assumption": weakest_assumption,
            "confidence": confidence if confidence in {"high", "medium", "low"} else "low",
            "errors": [] if valid else ["invalid boundary brief"],
        }

    @staticmethod
    def _normalize_boundary_jury(
        args: dict[str, Any],
        candidates: list[dict[str, Any]],
        available_lines: set[int],
    ) -> dict[str, Any]:
        winner_id = str(args.get("winner_id") or "unresolved")
        relations = {"same_speaker", "different_speaker", "unclear"}
        confidence = str(args.get("confidence") or "low")
        reason = str(args.get("reason") or "").strip()
        decisive = str(args.get("decisive_evidence") or "").strip()
        valid = bool(
            winner_id in {"A", "B", "unresolved"}
            and args.get("target_to_previous") in relations
            and args.get("target_to_next") in relations
            and confidence in {"high", "medium", "low"}
            and reason
            and decisive
        )
        winner_index = 0 if winner_id == "A" else 1 if winner_id == "B" else None
        return {
            "valid": valid,
            "status": "resolved" if valid and winner_index is not None else "unresolved",
            "winner_id": winner_id,
            "speaker": candidates[winner_index]["speaker"] if valid and winner_index is not None else "",
            "target_to_previous": args.get("target_to_previous", "unclear"),
            "target_to_next": args.get("target_to_next", "unclear"),
            "confidence": confidence if confidence in {"high", "medium", "low"} else "low",
            "citations": SceneSequenceAudit._int_list(args.get("citations"), available_lines),
            "decisive_evidence": decisive,
            "reason": reason,
            "errors": [] if valid else ["invalid boundary jury verdict"],
        }

    @staticmethod
    def _invalid_boundary_brief(hypothesis_id: str) -> dict[str, Any]:
        return {
            "valid": False,
            "hypothesis_id": hypothesis_id,
            "supporting_citations": [],
            "contradicting_citations": [],
            "strongest_case": "",
            "weakest_assumption": "missing structured brief",
            "confidence": "low",
            "errors": ["missing structured brief"],
        }

    @staticmethod
    def _invalid_boundary_jury(reason: str) -> dict[str, Any]:
        return {
            "valid": False,
            "status": "unresolved",
            "winner_id": "unresolved",
            "speaker": "",
            "target_to_previous": "unclear",
            "target_to_next": "unclear",
            "confidence": "low",
            "citations": [],
            "decisive_evidence": "",
            "reason": reason,
            "errors": [reason],
        }

    def _select_decision(
        self,
        baseline: str,
        baseline_evidence: dict[str, Any],
        scene_cards: list[dict[str, Any]],
        investigations: list[dict[str, Any]],
        verdict: dict[str, Any],
        packet: dict[str, Any],
        dialogue_index: int,
        local_evidence: dict[str, Any],
        contrastive_boundary: dict[str, Any] | None = None,
        baseline_reason: str = "",
    ) -> dict[str, Any]:
        baseline_clean = self.runtime.validate_speaker(baseline) or baseline
        proposed = self.runtime.validate_speaker(verdict.get("speaker", "")) or ""
        target_turn = next(
            turn for turn in packet["turns"] if turn["dialogue_index"] == dialogue_index
        )
        target_line = target_turn["line"]
        nearby_citation = any(
            abs(int(citation) - target_line) <= 8 for citation in verdict.get("citations", [])
        )
        adjacent_non_speech_ambiguity = self._has_adjacent_non_speech_mention(
            proposed,
            target_line,
            packet,
        )
        level = verdict.get("evidence_level")
        basis = verdict.get("evidence_basis")
        high = verdict.get("confidence") == "high"
        voice_kind = verdict.get("voice_kind", "unclear")
        verdict_quote_type = verdict.get("quote_type", "unclear")
        nonperson_quote_types = {"thought_or_narration", "sound_or_text"}
        strong_anchor = bool(
            verdict.get("status") == "resolved"
            and high
            and nearby_citation
            and voice_kind != "non_person"
            and verdict_quote_type not in nonperson_quote_types
            and (
                (
                    level == "direct"
                    and basis == "explicit_attribution"
                    and self._has_direct_speech_binding(
                        proposed, verdict.get("citations", []), packet, target_line
                    )
                )
                or (level == "identity" and basis == "identity_alias")
                or (level == "quote_type" and basis == "quote_type")
            )
        )
        decision = {
            "speaker": baseline_clean,
            "baseline_speaker": baseline_clean,
            "baseline_changed": False,
            "selection_source": "scene-evidence-baseline-retained",
            "evidence_gate": "insufficient-to-override",
            "confirmed_anchor": False,
            "voice_kind": baseline_evidence.get("voice_kind", "unclear"),
            "quote_type": baseline_evidence.get("quote_type", "unclear"),
            "evidence_basis": baseline_evidence.get("evidence_basis", "unknown"),
            "confidence": baseline_evidence.get("confidence", "low"),
            "citations": [],
        }
        scope_audit = self._baseline_scope_conflict(
            baseline_clean,
            baseline_evidence,
            baseline_reason,
            packet,
            dialogue_index,
            local_evidence,
        )
        decision["baseline_scope_audit"] = scope_audit

        person_candidate, person_scene_support = self._scene_person_consensus(scene_cards)
        partial_speech = self._has_partial_spoken_quote_container(target_line, packet)
        person_vocalization = bool(
            person_candidate
            and self._has_person_vocalization_binding(person_candidate, target_line, packet)
        )
        partial_speech_binding = bool(
            person_candidate
            and partial_speech
            and self._has_direct_speech_binding(person_candidate, [target_line], packet, target_line)
        )
        conveyed_quote = bool(
            person_candidate
            and self._has_conveyed_quote_binding(person_candidate, target_line, packet)
        )
        legacy_restoration = bool(
            self.allow_model_consensus_override
            and (partial_speech_binding or conveyed_quote)
        )
        if person_scene_support >= 2 and (person_vocalization or legacy_restoration):
            changed = not self.runtime.same_speaker(baseline_clean, person_candidate)
            if person_vocalization:
                source = "person-vocalization-restoration"
                gate = "bound-person-vocalization-plus-scene-consensus"
                restored_quote_type = "sound_or_text"
                restored_basis = "explicit_attribution"
            elif partial_speech_binding:
                source = "partial-speech-person-restoration"
                gate = "started-speech-plus-scene-consensus"
                restored_quote_type = "direct_speech"
                restored_basis = "explicit_attribution"
            else:
                source = "conveyed-quote-person-restoration"
                gate = "named-expression-plus-scene-consensus"
                restored_quote_type = "embedded_quote"
                restored_basis = "speaker_action"
            decision.update(
                speaker=person_candidate,
                baseline_changed=changed,
                selection_source=source,
                evidence_gate=gate,
                confirmed_anchor=True,
                voice_kind="person",
                quote_type=restored_quote_type,
                evidence_basis=restored_basis,
                confidence="high",
                citations=[target_line],
            )
            applied = self._apply_verdict_fields(decision, verdict)
            applied.update(
                voice_kind="person",
                quote_type=restored_quote_type,
                evidence_basis=restored_basis,
                confidence="high",
                citations=[target_line],
            )
            return applied

        matching_nonperson_investigations = [
            report for report in investigations
            if report.get("valid") and report.get("status") == "resolved"
            and report.get("voice_kind") == "non_person"
            and report.get("quote_type") in nonperson_quote_types
        ]
        scene_nonperson_support = sum(
            card.get("speaker") == self.runtime.non_person_label
            or card.get("quote_type") in nonperson_quote_types
            for card in scene_cards
            if card.get("valid")
        )
        deterministic_nonperson = self._has_nonperson_quote_container(target_line, packet)
        resolved_nonperson = bool(
            verdict.get("status") == "resolved"
            and proposed == self.runtime.non_person_label
            and voice_kind == "non_person"
            and verdict_quote_type in nonperson_quote_types
            and high
            and nearby_citation
        )
        if resolved_nonperson and deterministic_nonperson and matching_nonperson_investigations:
            changed = not self.runtime.same_speaker(baseline_clean, self.runtime.non_person_label)
            decision.update(
                speaker=self.runtime.non_person_label,
                baseline_changed=changed,
                selection_source="nonperson-container-override",
                evidence_gate="deterministic-unspoken-container",
                confirmed_anchor=True,
            )
            return self._apply_verdict_fields(decision, verdict)

        if not self.allow_model_consensus_override:
            if not self.runtime.is_valid_speaker(baseline_clean):
                if proposed and self.runtime.is_valid_speaker(proposed) and verdict.get("status") == "resolved":
                    decision.update(
                        speaker=proposed,
                        baseline_changed=True,
                        selection_source="invalid-baseline-scene-replacement",
                        evidence_gate="valid-scene-verdict",
                        confirmed_anchor=strong_anchor,
                    )
                else:
                    decision.update(
                        speaker="?",
                        selection_source="scene-unresolved-invalid-baseline",
                    )
                return self._apply_verdict_fields(decision, verdict)

            if scope_audit["conflict"]:
                repair = self._scope_repair_consensus(
                    baseline_clean,
                    scene_cards,
                    investigations,
                    target_line,
                )
                decision["scope_repair_consensus"] = repair
                contact_scope = bool(
                    _DIALOGUE_CONTACT_VERBS.intersection(scope_audit.get("consumed_verbs") or [])
                )
                verdict_matches_repair = bool(
                    repair.get("valid")
                    and proposed
                    and self.runtime.same_speaker(proposed, repair.get("speaker", ""))
                    and verdict.get("status") == "resolved"
                    and verdict.get("confidence") in {"high", "medium"}
                )
                if (
                    contact_scope
                    and verdict_matches_repair
                    and repair.get("support", 0) >= 3
                    and repair.get("contact_repair_support", 0) >= 2
                    and repair.get("speaker") != self.runtime.non_person_label
                ):
                    decision.update(
                        speaker=repair["speaker"],
                        baseline_changed=True,
                        selection_source="contact-scope-repair-override",
                        evidence_gate="consumed-contact-attribution-plus-cross-role-repair",
                        confirmed_anchor=False,
                        citations=repair["citations"],
                    )
                    return self._apply_verdict_fields(decision, verdict)

            if proposed and self.runtime.same_speaker(proposed, baseline_clean):
                decision["evidence_gate"] = "model-verdict-agrees-baseline-unconfirmed"
                decision["confirmed_anchor"] = False
            elif proposed:
                decision["rejected_scene_proposal"] = proposed
                decision["rejected_reason"] = (
                    "model consensus is retained as audit evidence but cannot override a valid baseline"
                )
            boundary = contrastive_boundary or {}
            boundary_speaker = self.runtime.validate_speaker(boundary.get("speaker", "")) or ""
            if boundary_speaker and not self.runtime.same_speaker(boundary_speaker, baseline_clean):
                decision["rejected_boundary_proposal"] = boundary_speaker
            decision["model_consensus_override_enabled"] = False
            return self._apply_verdict_fields(decision, verdict)

        boundary = contrastive_boundary or {}
        boundary_winner = self.runtime.validate_speaker(boundary.get("speaker", "")) or ""
        boundary_winner_ambiguity = bool(
            boundary_winner
            and self._has_adjacent_non_speech_mention(boundary_winner, target_line, packet)
        )
        boundary_deciders = boundary.get("consensus_deciders") or []
        boundary_scene_support = sum(
            self.runtime.same_speaker(card.get("speaker", ""), boundary_winner)
            for card in scene_cards
            if card.get("valid")
        )
        boundary_investigation_support = sum(
            report.get("valid")
            and report.get("status") == "resolved"
            and self.runtime.same_speaker(report.get("speaker", ""), boundary_winner)
            for report in investigations
        )
        boundary_source_backed = bool(
            boundary.get("consensus_source") == "initial_juries"
            or (
                boundary.get("consensus_source") == "disagreement_referees"
                and (
                    boundary_scene_support >= 3
                    or (boundary_scene_support >= 2 and boundary_investigation_support >= 2)
                )
            )
        )
        boundary_can_decide = bool(
            boundary.get("valid")
            and boundary.get("status") == "resolved"
            and boundary.get("unanimous")
            and boundary.get("confidence") in {"high", "medium"}
            and boundary_winner
            and self.runtime.is_valid_speaker(boundary_winner)
            and boundary_winner != self.runtime.non_person_label
            and len(boundary_deciders) >= 2
            and boundary_source_backed
            and all(
                jury.get("valid")
                and jury.get("status") == "resolved"
                and self.runtime.same_speaker(jury.get("speaker", ""), boundary_winner)
                for jury in boundary_deciders
            )
            and not deterministic_nonperson
            and not boundary_winner_ambiguity
            and not (strong_anchor and not self.runtime.same_speaker(proposed, boundary_winner))
        )
        if boundary_can_decide:
            changed = not self.runtime.same_speaker(boundary_winner, baseline_clean)
            decision.update(
                speaker=boundary_winner,
                baseline_changed=changed,
                selection_source=(
                    "contrastive-boundary-override"
                    if changed else "contrastive-boundary-baseline-confirmed"
                ),
                evidence_gate=(
                    "unanimous-mirrored-boundary-jury"
                    if boundary.get("consensus_source") == "initial_juries"
                    else "mirrored-referees-plus-scene-chain"
                ),
                confirmed_anchor=False,
                voice_kind="person",
                quote_type="direct_speech",
                evidence_basis="sequence_constraint",
                confidence=boundary.get("confidence", "medium"),
                citations=boundary.get("citations", []),
                verdict_status=verdict.get("status", "unresolved"),
                verdict_evidence_level=verdict.get("evidence_level", "insufficient"),
            )
            return decision

        scene_candidate_investigations = [
            report for report in investigations
            if report.get("valid") and report.get("status") == "resolved"
            and self.runtime.same_speaker(report.get("speaker", ""), person_candidate)
            and report.get("evidence_basis") not in {"alternation", "unknown"}
            and report.get("confidence") in {"high", "medium"}
        ]
        blind_scene_candidate_investigations = [
            report for report in scene_candidate_investigations
            if report.get("agent") in {
                "TargetOpenWorldInvestigator",
                "TargetSequenceBoundaryInvestigator",
                "TargetDelayedAttributionInvestigator",
            }
        ]
        scene_candidate_ambiguity = self._has_adjacent_non_speech_mention(
            person_candidate,
            target_line,
            packet,
        )
        valid_scene_cards = [card for card in scene_cards if card.get("valid")]
        supporting_scene_cards = [
            card for card in valid_scene_cards
            if self.runtime.same_speaker(card.get("speaker", ""), person_candidate)
        ]
        decoder_scene_support = any(
            card.get("agent") in {"SceneDialogueActMapper", "SceneReverseMapper"}
            for card in supporting_scene_cards
        )
        scene_chain_confirmed = bool(
            person_candidate
            and not self.runtime.same_speaker(person_candidate, baseline_clean)
            and (
                (len(valid_scene_cards) >= 4 and person_scene_support >= 4)
                or (
                    len(valid_scene_cards) >= 5
                    and person_scene_support >= 3
                    and decoder_scene_support
                )
                or
                (person_scene_support >= 4 and len(scene_candidate_investigations) >= 1)
                or (person_scene_support >= 3 and len(scene_candidate_investigations) >= 2)
                or (person_scene_support >= 3 and len(blind_scene_candidate_investigations) >= 1)
            )
            and not scene_candidate_ambiguity
            and not deterministic_nonperson
        )
        if scene_chain_confirmed:
            decision.update(
                speaker=person_candidate,
                baseline_changed=True,
                selection_source="scene-chain-conflict-override",
                evidence_gate=(
                    "four-of-five-scene-map-supermajority"
                    if (
                        len(valid_scene_cards) >= 4
                        and person_scene_support >= 4
                        and not scene_candidate_investigations
                    )
                    else "three-of-five-scene-decoder-majority"
                    if (
                        len(valid_scene_cards) >= 5
                        and person_scene_support >= 3
                        and decoder_scene_support
                        and not scene_candidate_investigations
                    )
                    else "four-scene-maps-plus-investigation"
                    if person_scene_support >= 4
                    else (
                        "three-scene-maps-plus-blind-investigation"
                        if blind_scene_candidate_investigations
                        else "three-scene-maps-plus-two-investigations"
                    )
                ),
                confirmed_anchor=False,
                voice_kind="person",
                quote_type="direct_speech",
                evidence_basis="sequence_constraint",
                confidence="high",
                citations=[target_line],
            )
            return decision

        scene_role_consensus = bool(
            len(valid_scene_cards) >= 3
            and person_candidate
            and self.runtime.is_generic_speaker(person_candidate)
            and all(
                self.runtime.same_speaker(card.get("speaker", ""), person_candidate)
                for card in valid_scene_cards
            )
        )
        if scene_role_consensus:
            identity_groups: list[dict[str, Any]] = []
            for report in investigations:
                candidate = self.runtime.validate_speaker(report.get("speaker", "")) or ""
                if (
                    not report.get("valid")
                    or report.get("status") != "resolved"
                    or not candidate
                    or self.runtime.is_generic_speaker(candidate)
                    or candidate == self.runtime.non_person_label
                    or report.get("evidence_basis") not in {"identity_alias", "explicit_attribution"}
                    or report.get("confidence") not in {"high", "medium"}
                    or not report.get("citations")
                ):
                    continue
                group = next(
                    (
                        item for item in identity_groups
                        if self.runtime.same_speaker(item["speaker"], candidate)
                    ),
                    None,
                )
                if group is None:
                    identity_groups.append({"speaker": candidate, "count": 1})
                else:
                    group["count"] += 1
            identity_groups.sort(key=lambda item: item["count"], reverse=True)
            if identity_groups and identity_groups[0]["count"] >= 2:
                named_speaker = identity_groups[0]["speaker"]
                decision.update(
                    speaker=named_speaker,
                    baseline_changed=not self.runtime.same_speaker(named_speaker, baseline_clean),
                    selection_source="scene-role-identity-canonicalization",
                    evidence_gate="unanimous-scene-role-plus-two-identity-investigations",
                    confirmed_anchor=True,
                    voice_kind="person",
                    quote_type="direct_speech",
                    evidence_basis="identity_alias",
                    confidence="high",
                    citations=[target_line],
                )
                return decision

        if not self.runtime.is_valid_speaker(baseline_clean):
            if proposed and self.runtime.is_valid_speaker(proposed) and verdict.get("status") == "resolved":
                decision.update(
                    speaker=proposed,
                    baseline_changed=True,
                    selection_source="invalid-baseline-scene-replacement",
                    evidence_gate="valid-scene-verdict",
                    confirmed_anchor=strong_anchor,
                )
            else:
                decision.update(speaker="?", selection_source="scene-unresolved-invalid-baseline")
            return self._apply_verdict_fields(decision, verdict)

        if (
            verdict.get("status") != "resolved"
            or not proposed
            or not self.runtime.is_valid_speaker(proposed)
            or self.runtime.same_speaker(proposed, baseline_clean)
        ):
            if proposed and self.runtime.same_speaker(proposed, baseline_clean):
                decision["evidence_gate"] = "verdict-confirmed-baseline"
                decision["confirmed_anchor"] = strong_anchor
            return self._apply_verdict_fields(decision, verdict)

        matching_investigations = [
            report for report in investigations
            if report.get("valid") and report.get("status") == "resolved"
            and self.runtime.same_speaker(report.get("speaker", ""), proposed)
        ]
        direct_investigations = [
            report for report in matching_investigations
            if report.get("evidence_basis") == "explicit_attribution"
            and report.get("confidence") in {"high", "medium"}
        ]
        quote_type_investigations = [
            report for report in matching_investigations
            if report.get("evidence_basis") == "quote_type"
            and report.get("confidence") in {"high", "medium"}
        ]
        identity_investigations = [
            report for report in matching_investigations
            if report.get("evidence_basis") == "identity_alias"
            and report.get("confidence") in {"high", "medium"}
        ]
        scene_support = sum(
            self.runtime.same_speaker(card.get("speaker", ""), proposed)
            for card in scene_cards
            if card.get("valid")
        )
        source = ""
        gate = ""
        confirmed = False
        if (
            level == "direct"
            and basis == "explicit_attribution"
            and high
            and nearby_citation
            and self._has_direct_speech_binding(
                proposed, verdict.get("citations", []), packet, target_line
            )
            and direct_investigations
        ):
            source = "direct-scene-evidence-override"
            gate = "direct-target-binding"
            confirmed = True
        elif (
            level == "quote_type"
            and basis == "quote_type"
            and high
            and nearby_citation
            and len(quote_type_investigations) >= 2
        ):
            source = "quote-type-scene-override"
            gate = "independent-quote-type-chain"
            confirmed = True
        elif (
            level == "identity"
            and basis == "identity_alias"
            and high
            and self.runtime.is_generic_speaker(baseline_clean)
            and not self.runtime.is_generic_speaker(proposed)
            and identity_investigations
        ):
            source = "identity-scene-override"
            gate = "verified-identity-chain"
            confirmed = True
        elif (
            high
            and level not in {"weak", "insufficient"}
            and basis not in {"alternation", "inference", "unknown"}
            and (
                (scene_support >= 3 and len(matching_investigations) >= 1)
                or (scene_support >= 2 and len(matching_investigations) >= 2)
                or len(matching_investigations) >= 3
            )
            and all(
                report.get("evidence_basis") not in {"alternation", "unknown"}
                and report.get("confidence") in {"high", "medium"}
                for report in matching_investigations
            )
            and nearby_citation
            and not adjacent_non_speech_ambiguity
        ):
            source = "corroborated-scene-sequence-override"
            if scene_support >= 3:
                gate = "unanimous-scene-plus-independent-investigation"
            elif scene_support >= 2:
                gate = "two-scene-maps-plus-two-investigations"
            else:
                gate = "three-independent-investigations"

        if source:
            decision.update(
                speaker=proposed,
                baseline_changed=True,
                selection_source=source,
                evidence_gate=gate,
                confirmed_anchor=confirmed,
            )
        else:
            decision["rejected_scene_proposal"] = proposed
            decision["rejected_reason"] = "proposal did not satisfy a source-backed override gate"
            decision["scene_support_for_rejected"] = scene_support
            decision["investigator_support_for_rejected"] = len(matching_investigations)
            decision["adjacent_non_speech_ambiguity"] = adjacent_non_speech_ambiguity
        return self._apply_verdict_fields(decision, verdict)

    def _target_scene_cards(
        self,
        scene_solution: dict[str, Any],
        dialogue_index: int,
    ) -> list[dict[str, Any]]:
        cards = []
        for scene_map in scene_solution["maps"]:
            assignment = assignment_for_dialogue(
                {"assignments": scene_map.get("assignments", {})},
                scene_solution["packet"],
                dialogue_index,
            )
            if assignment:
                card = dict(assignment)
                card["agent"] = scene_map.get("agent", "")
                card["valid"] = self.runtime.is_valid_speaker(card.get("speaker", ""))
                cards.append(card)
        return cards

    def _build_evidence_cases(
        self,
        baseline: str,
        baseline_evidence: dict[str, Any],
        baseline_reason: str,
        scene_cards: list[dict[str, Any]],
        local_evidence: dict[str, Any],
        line_num: int,
        *,
        include_baseline: bool = True,
    ) -> list[dict[str, Any]]:
        cases = []
        baseline_clean = self.runtime.validate_speaker(baseline) or baseline
        if include_baseline and baseline_clean:
            cases.append(
                {
                    "speaker": baseline_clean,
                    "evidence_basis": baseline_evidence.get("evidence_basis", "unknown"),
                    "confidence": baseline_evidence.get("confidence", "low"),
                    "citations": [line_num],
                    "reason": (baseline_reason or "provisional target analysis")[:1000],
                }
            )
        cases.extend(card for card in scene_cards if card.get("valid"))
        for item in (local_evidence or {}).get("attribution_candidates") or []:
            speaker = self.runtime.validate_speaker(item.get("speaker", ""))
            if not speaker:
                continue
            cases.append(
                {
                    "speaker": speaker,
                    "evidence_basis": "explicit_attribution",
                    "confidence": "medium",
                    "citations": [line_num],
                    "reason": f"local attribution parser found speech verb {item.get('verb') or '?'}",
                }
            )
        return cases

    def _neutralize_cases(
        self,
        cases: list[dict[str, Any]],
        line_num: int,
    ) -> list[dict[str, Any]]:
        best_by_speaker: dict[str, dict[str, Any]] = {}
        basis_rank = {
            "explicit_attribution": 6,
            "speaker_action": 5,
            "identity_alias": 5,
            "quote_type": 4,
            "sequence_constraint": 3,
            "inference": 2,
            "unknown": 1,
        }
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        for case in cases:
            speaker = self.runtime.validate_speaker(case.get("speaker", ""))
            if not speaker or not self.runtime.is_valid_speaker(speaker):
                continue
            candidate = dict(case)
            candidate["speaker"] = speaker
            score = (
                basis_rank.get(candidate.get("evidence_basis"), 0),
                confidence_rank.get(candidate.get("confidence"), 0),
                bool(candidate.get("citations")),
            )
            existing = best_by_speaker.get(speaker)
            if existing is None or score > existing["_score"]:
                candidate["_score"] = score
                best_by_speaker[speaker] = candidate
        ordered = sorted(
            best_by_speaker.values(),
            key=lambda case: (sum(ord(ch) for ch in case["speaker"]) + line_num) % 1009,
        )
        output = []
        for index, case in enumerate(ordered):
            case = {key: value for key, value in case.items() if key != "_score"}
            case["case_id"] = f"Case-{chr(ord('A') + index)}"
            output.append(case)
        return output

    def _neutralize_scene_sequences(
        self,
        scene_solution: dict[str, Any],
        line_num: int,
    ) -> list[dict[str, Any]]:
        packet = scene_solution["packet"]
        unique: dict[tuple[str, ...], dict[str, Any]] = {}
        for scene_map in scene_solution.get("maps") or []:
            if not scene_map.get("valid"):
                continue
            assignments = scene_map.get("assignments") or {}
            sequence = []
            signature = []
            for turn in packet["turns"]:
                item = assignments.get(turn["id"])
                if not item:
                    sequence = []
                    break
                speaker = self.runtime.validate_speaker(item.get("speaker", "")) or "?"
                signature.append(speaker)
                sequence.append(
                    {
                        "turn_id": turn["id"],
                        "line": turn["line"],
                        "speaker": speaker,
                        "evidence_basis": item.get("evidence_basis", "unknown"),
                        "confidence": item.get("confidence", "low"),
                        "citations": self._int_list(item.get("citations")),
                        "reason": str(item.get("reason") or "").strip()[:600],
                    }
                )
            if sequence:
                unique.setdefault(tuple(signature), {"assignments": sequence})

        ordered = sorted(
            unique.items(),
            key=lambda pair: (
                sum((index + 1) * sum(ord(ch) for ch in speaker) for index, speaker in enumerate(pair[0]))
                + line_num
            ) % 10007,
        )
        output = []
        for index, (_, sequence) in enumerate(ordered):
            output.append(
                {
                    "case_id": f"Sequence-{chr(ord('A') + index)}",
                    "assignments": sequence["assignments"],
                }
            )
        return output

    @staticmethod
    def _format_neutral_cases(cases: list[dict[str, Any]]) -> str:
        lines = [
            "[Unordered candidate evidence cases - no vote counts or source identities]",
            "The set may be incomplete and every case may be wrong.",
        ]
        for case in cases:
            lines.append(
                f"  {case['case_id']}: speaker={case['speaker']}; basis={case.get('evidence_basis')}; "
                f"confidence={case.get('confidence')}; citations={case.get('citations')}; "
                f"claim={case.get('reason')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_neutral_sequences(sequences: list[dict[str, Any]]) -> str:
        lines = [
            "[Deduplicated unordered full-scene hypotheses - no source identities or support counts]",
            "Each sequence is one distinct hypothesis; absence of duplicates does not imply weak support.",
        ]
        if not sequences:
            lines.append("  (no complete valid sequence hypothesis)")
        for sequence in sequences:
            lines.append(f"  {sequence['case_id']}:")
            for item in sequence["assignments"]:
                lines.append(
                    f"    {item['turn_id']} L{item['line']} -> {item['speaker']}; "
                    f"basis={item.get('evidence_basis')}; confidence={item.get('confidence')}; "
                    f"citations={item.get('citations')}; claim={item.get('reason')}"
                )
        return "\n".join(lines)

    def _normalize_target_card(
        self,
        args: dict[str, Any],
        available_lines: set[int],
    ) -> dict[str, Any]:
        status = str(args.get("status") or "unresolved").strip().lower()
        speaker = self.runtime.validate_speaker(args.get("speaker", "")) or ""
        voice_kind = str(args.get("voice_kind") or "unclear").strip().lower()
        citations = self._int_list(args.get("citations"))
        valid_citation = bool(citations and any(value in available_lines for value in citations))
        if status == "unresolved":
            speaker = ""
            voice_kind = "unclear"
            valid = True
        else:
            if voice_kind == "non_person":
                speaker = self.runtime.non_person_label
            valid = bool(self.runtime.is_valid_speaker(speaker) and valid_citation)
        return {
            "valid": valid,
            "status": status,
            "speaker": speaker,
            "voice_kind": voice_kind,
            "quote_type": str(args.get("quote_type") or "unclear").strip().lower(),
            "evidence_basis": str(args.get("evidence_basis") or "unknown").strip().lower(),
            "confidence": str(args.get("confidence") or "low").strip().lower(),
            "citations": citations,
            "reason": str(args.get("reason") or "").strip()[:2000],
            "rejected_candidates": [str(value) for value in args.get("rejected_candidates") or []][:12],
        }

    def _normalize_verdict(
        self,
        args: dict[str, Any],
        available_lines: set[int],
    ) -> dict[str, Any]:
        status = str(args.get("status") or "unresolved").strip().lower()
        speaker = self.runtime.validate_speaker(args.get("speaker", "")) or ""
        voice_kind = str(args.get("voice_kind") or "unclear").strip().lower()
        citations = self._int_list(args.get("citations"))
        valid_citation = bool(citations and any(value in available_lines for value in citations))
        if status == "unresolved":
            speaker = ""
            voice_kind = "unclear"
            valid = True
        else:
            if voice_kind == "non_person":
                speaker = self.runtime.non_person_label
            valid = bool(self.runtime.is_valid_speaker(speaker) and valid_citation)
        return {
            "valid": valid,
            "status": status,
            "speaker": speaker,
            "voice_kind": voice_kind,
            "quote_type": str(args.get("quote_type") or "unclear").strip().lower(),
            "evidence_level": str(args.get("evidence_level") or "insufficient").strip().lower(),
            "evidence_basis": str(args.get("evidence_basis") or "unknown").strip().lower(),
            "confidence": str(args.get("confidence") or "low").strip().lower(),
            "citations": citations,
            "reason": str(args.get("reason") or "").strip()[:2400],
            "contradicted_candidates": [
                str(value) for value in args.get("contradicted_candidates") or []
            ][:12],
        }

    def _compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.runtime.estimate_messages(messages) + self.runtime.max_output_tokens < self.runtime.context_limit - 2000:
            return messages
        compacted = []
        for message in messages:
            copied = dict(message)
            content = str(copied.get("content") or "")
            if copied.get("role") == "tool" and len(content) > 6000:
                copied["content"] = content[:3000] + "\n... [tool output compacted] ...\n" + content[-3000:]
            elif copied.get("role") == "assistant" and len(content) > 2000:
                copied["content"] = content[:2000]
            compacted.append(copied)
        prefix = compacted[:2]
        history = compacted[2:]
        groups = []
        index = 0
        while index < len(history):
            message = history[index]
            group = [message]
            index += 1
            if message.get("role") == "assistant" and message.get("tool_calls"):
                expected = len(message.get("tool_calls") or [])
                while index < len(history) and expected > 0 and history[index].get("role") == "tool":
                    group.append(history[index])
                    index += 1
                    expected -= 1
            elif message.get("role") == "tool":
                # A tool response without its assistant request is not valid API history.
                continue
            groups.append(group)

        while (
            len(groups) > 1
            and self.runtime.estimate_messages(prefix + [item for group in groups for item in group])
            + self.runtime.max_output_tokens
            >= self.runtime.context_limit - 2000
        ):
            groups.pop(0)
        return prefix + [item for group in groups for item in group]

    def _map_speaker(self, value: str) -> str | None:
        if (value or "").strip() == "?":
            return "?"
        speaker = self.runtime.validate_speaker(value)
        return speaker if speaker and self.runtime.is_valid_speaker(speaker) else None

    def _scene_person_consensus(
        self,
        scene_cards: list[dict[str, Any]],
    ) -> tuple[str, int]:
        groups: list[dict[str, Any]] = []
        for card in scene_cards:
            if not card.get("valid"):
                continue
            speaker = self.runtime.validate_speaker(card.get("speaker", "")) or ""
            if (
                not self.runtime.is_valid_speaker(speaker)
                or speaker == self.runtime.non_person_label
            ):
                continue
            group = next(
                (
                    item for item in groups
                    if self.runtime.same_speaker(item["speaker"], speaker)
                ),
                None,
            )
            if group is None:
                groups.append({"speaker": speaker, "count": 1})
            else:
                group["count"] += 1
        if not groups:
            return "", 0
        groups.sort(key=lambda item: item["count"], reverse=True)
        if len(groups) > 1 and groups[0]["count"] == groups[1]["count"]:
            return "", 0
        return groups[0]["speaker"], int(groups[0]["count"])

    @staticmethod
    def _has_direct_speech_binding(
        speaker: str,
        citations: Any,
        packet: dict[str, Any],
        target_line: int | None = None,
    ) -> bool:
        speaker = str(speaker or "").strip()
        if not speaker or speaker == "?":
            return False
        cited_lines = set(SceneSequenceAudit._int_list(citations))
        raw_by_line = {
            int(item["line"]): str(item.get("text") or "")
            for item in packet.get("raw_lines") or []
        }
        subject_pattern = re.compile(
            rf"(?:^|[。！？；，：\s])"
            rf"(?:这时|随后|接着|只见|却见|于是|而后|然后|但)?\s*"
            rf"{re.escape(speaker)}(?P<middle>[^。！？；]{{0,24}}?){_SPEECH_VERB_PATTERN}"
        )
        indirect_markers = re.compile(
            r"(?:不理会|听见|听到|看见|看着|望见|望着|盯着|等着)"
        )
        for line_num in cited_lines:
            if target_line is not None and abs(line_num - target_line) > 1:
                continue
            text = raw_by_line.get(line_num, "")
            for match in subject_pattern.finditer(text):
                if indirect_markers.search(match.group("middle") or ""):
                    continue
                if target_line is not None and line_num == target_line - 1:
                    previous_text = raw_by_line.get(line_num - 1, "")
                    has_previous_quote = bool(re.search(r"[「『“]", previous_text))
                    remainder = text[match.end():]
                    introduces_next_quote = bool(re.search(r"[:：]\s*$", remainder or text))
                    if has_previous_quote and not introduces_next_quote:
                        continue
                return True
        return False

    @staticmethod
    def _has_conveyed_quote_binding(
        speaker: str,
        target_line: int,
        packet: dict[str, Any],
    ) -> bool:
        speaker = str(speaker or "").strip()
        if not speaker or speaker == "?":
            return False
        text = next(
            (
                str(item.get("text") or "")
                for item in packet.get("raw_lines") or []
                if int(item.get("line", -1)) == target_line
            ),
            "",
        )
        if speaker not in text or not re.search(r"[「『“]", text):
            return False
        speaker_index = text.find(speaker)
        marker = _CONVEYED_QUOTE_PATTERN.search(text, speaker_index + len(speaker))
        expression = _CONVEYED_EXPRESSION_PATTERN.search(text, speaker_index + len(speaker))
        return bool(
            (marker and marker.start() - speaker_index <= 72)
            or (expression and expression.start() - speaker_index <= 72)
        )

    @staticmethod
    def _has_adjacent_non_speech_mention(
        speaker: str,
        target_line: int,
        packet: dict[str, Any],
    ) -> bool:
        speaker = str(speaker or "").strip()
        if not speaker or speaker == "?":
            return False
        for item in packet.get("raw_lines") or []:
            try:
                line_num = int(item["line"])
            except (KeyError, TypeError, ValueError):
                continue
            if line_num not in {target_line - 1, target_line}:
                continue
            text = str(item.get("text") or "")
            if (
                speaker in text
                and not SceneSequenceAudit._has_direct_speech_binding(
                    speaker,
                    [line_num],
                    packet,
                    target_line,
                )
            ):
                return True
        return False

    @staticmethod
    def _has_nonperson_quote_container(
        target_line: int,
        packet: dict[str, Any],
    ) -> bool:
        for item in packet.get("raw_lines") or []:
            try:
                line_num = int(item["line"])
            except (KeyError, TypeError, ValueError):
                continue
            if line_num == target_line and _NONPERSON_CONTAINER_RE.search(
                str(item.get("text") or "")
            ):
                return True
        return False

    @staticmethod
    def _has_partial_spoken_quote_container(
        target_line: int,
        packet: dict[str, Any],
    ) -> bool:
        for item in packet.get("raw_lines") or []:
            try:
                line_num = int(item["line"])
            except (KeyError, TypeError, ValueError):
                continue
            if line_num == target_line and _PARTIAL_SPEECH_CONTAINER_RE.search(
                str(item.get("text") or "")
            ):
                return True
        return False

    @staticmethod
    def _has_person_vocalization_binding(
        speaker: str,
        target_line: int,
        packet: dict[str, Any],
    ) -> bool:
        speaker = str(speaker or "").strip()
        if not speaker:
            return False
        pattern = re.compile(
            rf"{re.escape(speaker)}[^。！？]{{0,100}}{_PERSON_VOCALIZATION_PATTERN}"
        )
        for item in packet.get("raw_lines") or []:
            try:
                line_num = int(item["line"])
            except (KeyError, TypeError, ValueError):
                continue
            if line_num == target_line and pattern.search(str(item.get("text") or "")):
                return True
        return False

    @staticmethod
    def _int_list(values: Any, allowed: set[int] | None = None) -> list[int]:
        output = []
        for value in values or []:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if allowed is not None and number not in allowed:
                continue
            if number not in output:
                output.append(number)
        return output

    @staticmethod
    def _unresolved_card(reason: str) -> dict[str, Any]:
        return {
            "valid": True,
            "status": "unresolved",
            "speaker": "",
            "voice_kind": "unclear",
            "quote_type": "unclear",
            "evidence_basis": "unknown",
            "confidence": "low",
            "citations": [],
            "reason": reason,
            "rejected_candidates": [],
        }

    @staticmethod
    def _unresolved_verdict(reason: str) -> dict[str, Any]:
        return {
            "valid": True,
            "status": "unresolved",
            "speaker": "",
            "voice_kind": "unclear",
            "quote_type": "unclear",
            "evidence_level": "insufficient",
            "evidence_basis": "unknown",
            "confidence": "low",
            "citations": [],
            "reason": reason,
            "contradicted_candidates": [],
        }

    @staticmethod
    def _apply_verdict_fields(
        decision: dict[str, Any],
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        decision["quote_type"] = verdict.get("quote_type", decision.get("quote_type", "unclear"))
        decision["voice_kind"] = verdict.get("voice_kind", decision.get("voice_kind", "unclear"))
        decision["evidence_basis"] = verdict.get("evidence_basis", decision.get("evidence_basis", "unknown"))
        decision["confidence"] = verdict.get("confidence", decision.get("confidence", "low"))
        decision["citations"] = verdict.get("citations", [])
        decision["verdict_status"] = verdict.get("status", "unresolved")
        decision["verdict_evidence_level"] = verdict.get("evidence_level", "insufficient")
        return decision
