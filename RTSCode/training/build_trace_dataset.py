from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from utils.dataset import load_clean_dataset, build_perturbed_dataset
from utils.inspector import Inspector, InspectorConfig
from utils.judges import EliminationJudge, JudgeConfig
from utils.llm_client import HFLocalClient, OpenAIClient
from utils.logging_utils import get_logger
from utils.io_utils import write_jsonl
from utils.perturbation import PerturbationConfig
from utils.trace_schema import TraceParseError, trace_output_consistent


def make_llm_client(kind: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    if kind == 'openai':
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url)
    if kind == 'hf_local':
        return HFLocalClient(model_name_or_path=model)
    raise ValueError(f'Unknown llm kind {kind}')


def main(args):
    logger = get_logger('build_trace_dataset', log_file=args.log_file)

    clean = load_clean_dataset(
        args.input_jsonl,
        question_key=args.question_key,
        answer_key=args.answer_key,
        premise_key=args.premise_key,
        limit=args.limit,
    )
    logger.info(f'Loaded clean instances: {len(clean)}')

    pert_cfg = PerturbationConfig(
        seed=args.seed,
        strength=args.strength,
        redundancy_prob=args.redundancy_prob,
        distractor_prob=args.distractor_prob,
        error_prob=args.error_prob,
        disorder_prob=args.disorder_prob,
    )
    perturbed = build_perturbed_dataset(clean, pert_cfg)
    logger.info(f'Built perturbed instances: {len(perturbed)}')

    llm = make_llm_client(args.aux_llm_kind, args.aux_llm_model, api_key=args.openai_api_key, base_url=args.openai_base_url)
    inspector = Inspector(llm, InspectorConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature))

    judge = EliminationJudge(JudgeConfig(), llm=None if args.judge_kind == 'heuristic' else llm)

    out_rows: List[Dict[str, Any]] = []
    skipped = 0

    for item in perturbed:
        q = item['question']
        P = item['premises_perturbed']

        try:
            ops, inspected, raw = inspector.inspect(q, P)
        except TraceParseError as e:
            skipped += 1
            logger.warning(f"Skip id {item['id']} parse error: {e}")
            continue
        except Exception as e:
            skipped += 1
            logger.warning(f"Skip id {item['id']} inspect error: {e}")
            continue

        if not trace_output_consistent(P, ops, inspected):
            skipped += 1
            logger.warning(f"Skip id {item['id']} trace output mismatch")
            continue

        ok = judge.judge(q, P, inspected)
        if not ok:
            skipped += 1
            logger.warning(f"Skip id {item['id']} judge reject")
            continue

        row = {
            'id': item['id'],
            'question': q,
            'premises': P,
            'trace': [op.to_dict() for op in ops],
            'inspected_premises': inspected,
            'answer': item['answer'],
            'perturbations': item.get('perturbations', []),
            'raw_model_output': raw,
        }
        out_rows.append(row)

        if args.save_every > 0 and len(out_rows) % args.save_every == 0:
            logger.info(f'Processed {len(out_rows)} kept, {skipped} skipped')

    logger.info(f'Finished. Kept {len(out_rows)}, skipped {skipped}')
    write_jsonl(args.output_jsonl, out_rows, append=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--question_key', type=str, default='question')
    parser.add_argument('--answer_key', type=str, default='answer')
    parser.add_argument('--premise_key', type=str, default='premises')
    parser.add_argument('--limit', type=int, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strength', type=float, default=0.6)
    parser.add_argument('--redundancy_prob', type=float, default=0.6)
    parser.add_argument('--distractor_prob', type=float, default=0.6)
    parser.add_argument('--error_prob', type=float, default=0.3)
    parser.add_argument('--disorder_prob', type=float, default=0.6)

    parser.add_argument('--aux_llm_kind', type=str, default='openai', choices=['openai', 'hf_local'])
    parser.add_argument('--aux_llm_model', type=str, required=True)
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--openai_base_url', type=str, default=None)

    parser.add_argument('--judge_kind', type=str, default='heuristic', choices=['heuristic', 'llm'])

    parser.add_argument('--max_new_tokens', type=int, default=900)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--log_file', type=str, default=None)

    args = parser.parse_args()
    main(args)
