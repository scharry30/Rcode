from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .random_utils import set_seed


@dataclass
class PerturbationConfig:
    seed: int = 42
    strength: float = 0.5
    redundancy_prob: float = 0.5
    distractor_prob: float = 0.5
    error_prob: float = 0.3
    disorder_prob: float = 0.5
    max_duplicates: int = 2
    max_distractors: int = 3


DISTRACTOR_POOL = [
    'Walking is healthy.',
    'Cars are fast.',
    'Time is money.',
    'Some statements may be irrelevant.',
    'It is sunny today.',
    'People usually prefer convenience.',
    'In general, trains are comfortable.',
]


def _paraphrase_like(p: str) -> str:
    # Lightweight rule based paraphrase to increase redundancy
    p2 = p.strip()
    p2 = re.sub(r'\s+', ' ', p2)
    # swap some connectors
    p2 = p2.replace('and', 'as well as') if 'and' in p2 else p2
    if not p2.endswith('.'):
        p2 += '.'
    return p2


def add_redundancy(premises: List[str], cfg: PerturbationConfig) -> List[str]:
    out = list(premises)
    k = max(1, int(cfg.max_duplicates * cfg.strength))
    for _ in range(k):
        if not out:
            break
        idx = random.randint(0, len(out) - 1)
        base = out[idx]
        if random.random() < 0.5:
            out.insert(idx, base)
        else:
            out.insert(idx, _paraphrase_like(base))
    return out


def add_distractors(premises: List[str], cfg: PerturbationConfig) -> List[str]:
    out = list(premises)
    k = max(1, int(cfg.max_distractors * cfg.strength))
    for _ in range(k):
        out.append(random.choice(DISTRACTOR_POOL))
    return out


def add_errors(premises: List[str], cfg: PerturbationConfig) -> List[str]:
    # Simple contradiction injection: flip a numeric value if present
    out = list(premises)
    if not out:
        return out
    idx = random.randint(0, len(out) - 1)
    p = out[idx]
    nums = re.findall(r'\d+(?:\.\d+)?', p)
    if nums:
        n = nums[0]
        try:
            val = float(n)
            # flip by an offset scaled by strength
            delta = max(1.0, 3.0 * cfg.strength)
            new_val = val + delta if random.random() < 0.5 else max(0.0, val - delta)
            if int(new_val) == new_val:
                new_val_str = str(int(new_val))
            else:
                new_val_str = str(new_val)
            out[idx] = p.replace(n, new_val_str, 1)
        except Exception:
            out[idx] = 'This premise may contain an error.'
    else:
        out[idx] = 'This premise may contain an error.'
    return out


def disorder(premises: List[str], cfg: PerturbationConfig) -> List[str]:
    out = list(premises)
    random.shuffle(out)
    return out


def perturb_premises(premises: List[str], cfg: PerturbationConfig) -> Tuple[List[str], List[str]]:
    """Return perturbed premises and a list of applied perturbation types."""
    set_seed(cfg.seed, deterministic=False)
    out = list(premises)
    applied: List[str] = []

    if random.random() < cfg.redundancy_prob:
        out = add_redundancy(out, cfg)
        applied.append('redundancy')

    if random.random() < cfg.distractor_prob:
        out = add_distractors(out, cfg)
        applied.append('distractor')

    if random.random() < cfg.error_prob:
        out = add_errors(out, cfg)
        applied.append('error')

    if random.random() < cfg.disorder_prob:
        out = disorder(out, cfg)
        applied.append('disorder')

    return out, applied
