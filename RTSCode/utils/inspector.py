from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .llm_client import ChatMessage, LLMClient
from .prompts import DEFAULT_INSPECT_SYSTEM, build_inspection_user_prompt
from .trace_schema import TraceOp, parse_rts_output


@dataclass
class InspectorConfig:
    system_prompt: str = DEFAULT_INSPECT_SYSTEM
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0


class Inspector:
    def __init__(self, llm: LLMClient, cfg: InspectorConfig):
        self.llm = llm
        self.cfg = cfg

    def inspect(self, question: str, premises: List[str]) -> Tuple[List[TraceOp], List[str], str]:
        user_prompt = build_inspection_user_prompt(question, premises)
        messages = [
            ChatMessage(role='system', content=self.cfg.system_prompt),
            ChatMessage(role='user', content=user_prompt),
        ]
        raw = self.llm.chat(
            messages,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        ops, inspected = parse_rts_output(raw)
        return ops, inspected, raw
