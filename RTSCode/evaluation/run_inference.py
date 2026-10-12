from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from utils.io_utils import read_jsonl, write_jsonl
from utils.inspector import Inspector, InspectorConfig
from utils.llm_client import HFLocalClient, OpenAIClient
from utils.logging_utils import get_logger
from utils.solver import Solver, SolverConfig
from utils.judges import EliminationJudge, JudgeConfig
from utils.trace_schema import execute_trace
from utils.metrics import score_prediction


def make_client(kind: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    if kind == 'hf_local':
        return HFLocalClient(model)
    if kind == 'openai':
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url)
    raise ValueError(f'unknown kind {kind}')


def main(args):
    logger = get_logger('rts_infer', log_file=args.log_file)

    items = read_jsonl(args.input_jsonl, limit=args.limit)
    logger.info(f'Loaded {len(items)} examples')

    inspector_client = make_client(args.inspector_kind, args.inspector_model, api_key=args.openai_api_key, base_url=args.openai_base_url)
    solver_client = make_client(args.solver_kind, args.solver_model, api_key=args.openai_api_key, base_url=args.openai_base_url)

    inspector = Inspector(inspector_client, InspectorConfig(max_new_tokens=args.inspect_max_new_tokens, temperature=args.inspect_temperature))
    solver = Solver(solver_client, SolverConfig(max_new_tokens=args.solve_max_new_tokens, temperature=args.solve_temperature))
    judge = EliminationJudge(JudgeConfig(), llm=None)

    outputs: List[Dict[str, Any]] = []
    for i, ex in enumerate(items):
        q = ex.get('question') or ex.get('problem') or ex.get('prompt')
        if isinstance(q, dict) and 'content' in q:
            q = q['content']
        q = str(q)
        premises = ex.get('premises') or ex.get('premises_perturbed') or ex.get('premises_clean') or ex.get('context')
        if isinstance(premises, str):
            premises = [premises]
        premises = [str(p).strip() for p in premises]
        gold = str(ex.get('answer', ex.get('ground_truth', '')))

        try:
            ops, inspected, raw_inspect = inspector.inspect(q, premises)
            inspected = execute_trace(premises, ops)
            inspect_ok = True
        except Exception as e:
            raw_inspect = str(e)
            inspected = []
            inspect_ok = False

        if inspected:
            solve_out = solver.solve(q, inspected)
            score = score_prediction(gold, solve_out, method=args.answer_verify)
            elim = judge.judge(q, premises, inspected)
        else:
            solve_out = ''
            score = score_prediction(gold, solve_out, method=args.answer_verify)
            elim = False

        outputs.append({
            'id': ex.get('id', i),
            'question': q,
            'premises': premises,
            'inspect_ok': inspect_ok,
            'inspected_premises': inspected,
            'raw_inspection': raw_inspect,
            'solver_output': solve_out,
            'pred': score.pred,
            'gold': score.gold,
            'correct': score.correct,
            'elim_ok': bool(elim),
        })

        if (i + 1) % args.logging_steps == 0:
            logger.info(f'processed {i+1} examples')

    write_jsonl(args.output_jsonl, outputs, append=False)
    logger.info(f'Saved to {args.output_jsonl}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)

    parser.add_argument('--inspector_kind', type=str, default='hf_local', choices=['hf_local', 'openai'])
    parser.add_argument('--inspector_model', type=str, required=True)
    parser.add_argument('--solver_kind', type=str, default='hf_local', choices=['hf_local', 'openai'])
    parser.add_argument('--solver_model', type=str, required=True)

    parser.add_argument('--inspect_max_new_tokens', type=int, default=900)
    parser.add_argument('--inspect_temperature', type=float, default=0.0)
    parser.add_argument('--solve_max_new_tokens', type=int, default=256)
    parser.add_argument('--solve_temperature', type=float, default=0.0)

    parser.add_argument('--answer_verify', type=str, default='string', choices=['string', 'float', 'auto'])

    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--openai_base_url', type=str, default=None)

    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--log_file', type=str, default=None)

    args = parser.parse_args()
    main(args)
