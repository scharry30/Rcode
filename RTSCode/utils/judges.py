from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .llm_client import ChatMessage, LLMClient
from .trace_schema import normalize_premise


def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r'[a-z0-9]+', a.lower()))
    sb = set(re.findall(r'[a-z0-9]+', b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


@dataclass
class JudgeConfig:
    duplicate_sim: float = 0.85
    min_length: int = 1
    allow_error_fixes: bool = True


class EliminationJudge:
    """Judge that checks whether inspected premises reduce common perturbations.

    This is used for R_elim in RL. In the paper, a separate auxiliary LLM judge can be used.
    This implementation provides a deterministic heuristic score, and optionally defers to an LLM client.
    """

    def __init__(self, cfg: JudgeConfig, llm: Optional[LLMClient] = None):
        self.cfg = cfg
        self.llm = llm

    def _heuristic(self, original: List[str], inspected: List[str]) -> bool:
        if len(inspected) < self.cfg.min_length:
            return False

        # Heuristic 1: duplicates reduced
        def count_duplicates(ps: List[str]) -> int:
            norm = [normalize_premise(p) for p in ps]
            dup = 0
            for i in range(len(norm)):
                for j in range(i + 1, len(norm)):
                    if _jaccard(norm[i], norm[j]) >= self.cfg.duplicate_sim:
                        dup += 1
            return dup

        orig_dup = count_duplicates(original)
        insp_dup = count_duplicates(inspected)
        dup_ok = insp_dup <= max(0, orig_dup - 1) if orig_dup > 0 else True

        # Heuristic 2: common distractor phrases removed
        distractor_markers = [
            'walking is healthy',
            'cars are fast',
            'time is money',
            'sunny today',
            'irrelevant',
            'in general',
        ]
        orig_d = sum(1 for p in original for m in distractor_markers if m in p.lower())
        insp_d = sum(1 for p in inspected for m in distractor_markers if m in p.lower())
        dist_ok = insp_d <= max(0, orig_d - 1) if orig_d > 0 else True

        # Heuristic 3: empty or placeholder premises reduced
        bad_pat = re.compile(r'(may contain an error|unknown|irrelevant)', re.I)
        orig_bad = sum(1 for p in original if bad_pat.search(p))
        insp_bad = sum(1 for p in inspected if bad_pat.search(p))
        bad_ok = insp_bad <= orig_bad

        return dup_ok and dist_ok and bad_ok

    def judge(self, question: str, original: List[str], inspected: List[str]) -> bool:
        if self.llm is None:
            return self._heuristic(original, inspected)

        prompt = (
            'You are evaluating whether the inspected premises remove redundancy, distractors, errors, and disorder. '
            'Return only YES or NO.\n\n'
            f'Question: {question}\n\n'
            'Original premises:\n' + '\n'.join([f'{i+1}. {p}' for i, p in enumerate(original)]) + '\n\n'
            'Inspected premises:\n' + '\n'.join([f'{i+1}. {p}' for i, p in enumerate(inspected)])
        )
        resp = self.llm.chat([ChatMessage(role='user', content=prompt)], max_new_tokens=8, temperature=0.0)
        return resp.strip().upper().startswith('Y')
