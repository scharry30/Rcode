from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_client import ChatMessage, LLMClient, HFLocalClient
from .prompts import DEFAULT_SOLVE_SYSTEM, build_solver_user_prompt


@dataclass
class SolverConfig:
    system_prompt: str = DEFAULT_SOLVE_SYSTEM
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


class Solver:
    def __init__(self, llm: LLMClient, cfg: SolverConfig):
        self.llm = llm
        self.cfg = cfg

    def solve(self, question: str, inspected_premises: list[str]) -> str:
        user_prompt = build_solver_user_prompt(question, inspected_premises)
        messages = [
            ChatMessage(role='system', content=self.cfg.system_prompt),
            ChatMessage(role='user', content=user_prompt),
        ]
        return self.llm.chat(
            messages,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
