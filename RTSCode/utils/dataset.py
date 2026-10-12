from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from .io_utils import read_jsonl
from .perturbation import PerturbationConfig, perturb_premises


def extract_premises(example: Dict[str, Any], premise_key: str = 'premises') -> List[str]:
    if premise_key in example and isinstance(example[premise_key], list):
        return [str(x).strip() for x in example[premise_key] if str(x).strip()]
    # fallback: split question context by sentence
    text = str(example.get('context', ''))
    if not text:
        text = str(example.get('question', ''))
    # simple sentence split
    sents = re.split(r'(?<=[\.!\?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


@dataclass
class RTSExample:
    question: str
    premises: List[str]
    answer: str
    meta: Optional[Dict[str, Any]] = None


def load_clean_dataset(path: str, question_key: str = 'question', answer_key: str = 'answer', premise_key: str = 'premises', limit: Optional[int] = None) -> List[RTSExample]:
    raw = read_jsonl(path, limit=limit)
    out: List[RTSExample] = []
    for ex in raw:
        q = str(ex.get(question_key, '')).strip()
        a = str(ex.get(answer_key, '')).strip()
        p = extract_premises(ex, premise_key=premise_key)
        if q and a:
            out.append(RTSExample(question=q, premises=p, answer=a, meta=ex))
    return out


def build_perturbed_dataset(examples: List[RTSExample], cfg: PerturbationConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        pcfg = PerturbationConfig(**cfg.__dict__)
        pcfg.seed = cfg.seed + i
        prem, applied = perturb_premises(ex.premises, pcfg)
        out.append({
            'id': i,
            'question': ex.question,
            'premises_clean': ex.premises,
            'premises_perturbed': prem,
            'answer': ex.answer,
            'perturbations': applied,
        })
    return out


def to_hf_dataset(items: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(items)
