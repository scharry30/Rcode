from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from utils.dataset import load_clean_dataset, build_perturbed_dataset
from utils.perturbation import PerturbationConfig
from utils.io_utils import write_jsonl
from utils.logging_utils import get_logger


def main(args):
    logger = get_logger('rts_robustness', log_file=args.log_file)
    clean = load_clean_dataset(args.clean_jsonl, question_key=args.question_key, answer_key=args.answer_key, premise_key=args.premise_key, limit=args.limit)
    logger.info(f'Loaded clean {len(clean)}')

    strengths = [float(s) for s in args.strengths.split(',') if s.strip()]
    out_all: List[Dict[str, Any]] = []

    for st in strengths:
        cfg = PerturbationConfig(seed=args.seed, strength=st, redundancy_prob=args.redundancy_prob, distractor_prob=args.distractor_prob, error_prob=args.error_prob, disorder_prob=args.disorder_prob)
        pert = build_perturbed_dataset(clean, cfg)
        for row in pert:
            row['strength'] = st
        out_all.extend(pert)
        logger.info(f'generated {len(pert)} examples at strength {st}')

    write_jsonl(args.output_jsonl, out_all, append=False)
    logger.info(f'Saved robustness set to {args.output_jsonl}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--question_key', type=str, default='question')
    parser.add_argument('--answer_key', type=str, default='answer')
    parser.add_argument('--premise_key', type=str, default='premises')
    parser.add_argument('--limit', type=int, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strengths', type=str, default='0.2,0.5,0.8')
    parser.add_argument('--redundancy_prob', type=float, default=0.6)
    parser.add_argument('--distractor_prob', type=float, default=0.6)
    parser.add_argument('--error_prob', type=float, default=0.3)
    parser.add_argument('--disorder_prob', type=float, default=0.6)

    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
