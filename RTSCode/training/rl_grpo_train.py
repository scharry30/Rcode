from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io_utils import read_jsonl
from utils.logging_utils import get_logger
from utils.prompts import build_inspection_user_prompt, DEFAULT_INSPECT_SYSTEM
from utils.trace_schema import TraceParseError, parse_rts_output, TRACE_START, TRACE_END
from utils.trace_schema import execute_trace
from utils.judges import EliminationJudge, JudgeConfig
from utils.metrics import score_prediction
from utils.solver import Solver, SolverConfig
from utils.llm_client import HFLocalClient


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 900
    temperature: float = 0.8
    top_p: float = 0.95

    clip_range: float = 0.2
    beta_kl: float = 0.02

    lr: float = 1e-6
    batch_size: int = 1
    grad_accum: int = 8
    num_steps: int = 2000

    save_every: int = 200


def _token_ids(tokenizer: AutoTokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)['input_ids']


def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle:
        return None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def build_prompt(question: str, premises: List[str]) -> str:
    user = build_inspection_user_prompt(question, premises)
    return f'SYSTEM: {DEFAULT_INSPECT_SYSTEM}\nUSER: {user}\nASSISTANT: '


def extract_trace_token_mask(tokenizer: AutoTokenizer, full_ids: List[int]) -> torch.Tensor:
    """Return a boolean mask over positions (excluding position 0) marking tokens that belong to <trace> segment.

    The mask aligns with action tokens, which correspond to logits positions 0..L-2 predicting ids 1..L-1.
    """
    start_ids = _token_ids(tokenizer, TRACE_START)
    end_ids = _token_ids(tokenizer, TRACE_END)

    s = find_subsequence(full_ids, start_ids)
    e = find_subsequence(full_ids, end_ids)

    mask = torch.zeros(len(full_ids), dtype=torch.bool)
    if s is None or e is None or e <= s:
        return mask

    # include the content between tags but exclude the start tag itself
    begin = s + len(start_ids)
    end = e
    if begin < end:
        mask[begin:end] = True
    return mask


def logprobs_for_ids(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Compute token logprobs for chosen ids.

    logits: [B, L, V]
    ids: [B, L]

    Returns logprobs aligned to ids positions, where position 0 has no prediction and is set to 0.
    """
    logp = torch.log_softmax(logits, dim=-1)
    # shift so logits at t predict token at t+1
    logp_next = logp[:, :-1, :]
    ids_next = ids[:, 1:]
    chosen = torch.gather(logp_next, dim=-1, index=ids_next.unsqueeze(-1)).squeeze(-1)
    pad = torch.zeros((ids.shape[0], 1), device=chosen.device, dtype=chosen.dtype)
    return torch.cat([pad, chosen], dim=1)


def compute_kl_sampled(logp_new: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sampled reverse KL estimator on chosen tokens.

    For a sampled sequence a, a common PPO style KL term uses (logp_new - logp_ref) on chosen actions.
    """
    diff = (logp_new - logp_ref) * mask
    denom = mask.sum().clamp(min=1)
    return diff.sum() / denom


def main(args):
    logger = get_logger('rts_grpo', log_file=args.log_file)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)

    cfg = GRPOConfig(
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_range=args.clip_range,
        beta_kl=args.beta_kl,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_steps=args.num_steps,
        save_every=args.save_every,
    )

    rows = read_jsonl(args.train_jsonl, limit=args.limit)
    if not rows:
        raise ValueError('Empty training set')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=None, torch_dtype=torch.bfloat16 if args.bf16 else None)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=None, torch_dtype=torch.bfloat16 if args.bf16 else None)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model, optimizer = accelerator.prepare(model, optimizer)
    ref_model = accelerator.prepare(ref_model)

    judge = EliminationJudge(JudgeConfig(), llm=None)

    solver_client = HFLocalClient(args.solver_model_name_or_path)
    solver = Solver(solver_client, SolverConfig(max_new_tokens=args.solver_max_new_tokens, temperature=0.0))

    start_tag_ids = _token_ids(tokenizer, TRACE_START)
    end_tag_ids = _token_ids(tokenizer, TRACE_END)

    def sample_group(prompt_ids: torch.Tensor) -> torch.Tensor:
        gen = accelerator.unwrap_model(model).generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return gen

    global_step = 0
    model.train()

    for step in range(cfg.num_steps):
        row = rows[step % len(rows)]
        question = row['question']
        premises = row.get('premises', row.get('premises_perturbed', row.get('premises_clean')))
        gold = row['answer']

        prompt = build_prompt(question, premises)
        prompt_enc = tokenizer(prompt, return_tensors='pt')
        prompt_ids = prompt_enc['input_ids'].to(accelerator.device)

        # Sample a group of candidates
        cand_ids: List[torch.Tensor] = []
        cand_text: List[str] = []
        inspected_lists: List[List[str]] = []
        valid_mask: List[bool] = []
        rewards: List[float] = []

        with torch.no_grad():
            for _ in range(cfg.group_size):
                gen = sample_group(prompt_ids)
                full_ids = gen[0]
                text = tokenizer.decode(full_ids[prompt_ids.shape[1]:], skip_special_tokens=True)
                full_text = text.strip()

                # Parse and execute inspection
                try:
                    ops, inspected = parse_rts_output(full_text)
                    inspected_exec = execute_trace(premises, ops)
                    inspected = inspected_exec
                    parse_ok = True
                except Exception:
                    inspected = []
                    parse_ok = False

                # Rewards
                if parse_ok and inspected:
                    solve_out = solver.solve(question, inspected)
                    solve_ok = float(score_prediction(gold, solve_out, method=args.answer_verify).correct)
                    elim_ok = float(judge.judge(question, premises, inspected))
                    r = solve_ok + elim_ok
                else:
                    r = 0.0

                cand_ids.append(full_ids)
                cand_text.append(full_text)
                inspected_lists.append(inspected)
                valid_mask.append(parse_ok)
                rewards.append(r)

        # Normalize advantages within group
        r_t = torch.tensor(rewards, device=accelerator.device, dtype=torch.float32)
        if r_t.std().item() < 1e-6:
            adv = r_t - r_t.mean()
        else:
            adv = (r_t - r_t.mean()) / (r_t.std() + 1e-6)

        # Policy optimization
        loss_accum = 0.0
        kl_accum = 0.0

        # Snapshot the current policy as "old" for this group.
        # This mirrors pi_{theta_old} in the paper and avoids copying per candidate.
        with torch.no_grad():
            old_model = copy.deepcopy(accelerator.unwrap_model(model)).to(accelerator.device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        for g in range(cfg.group_size):
            ids = cand_ids[g].unsqueeze(0).to(accelerator.device)

            logits_new = model(ids).logits
            with torch.no_grad():
                logits_old = old_model(ids).logits
                logits_ref = ref_model(ids).logits

            logp_new = logprobs_for_ids(logits_new, ids)
            logp_old = logprobs_for_ids(logits_old, ids)
            logp_ref = logprobs_for_ids(logits_ref, ids)

            token_mask = extract_trace_token_mask(tokenizer, ids[0].tolist()).to(accelerator.device)

            # Skip if no trace span found
            if token_mask.sum().item() == 0:
                continue

            ratio = torch.exp((logp_new - logp_old))
            a = adv[g].detach()

            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * a
            obj = torch.minimum(unclipped, clipped)

            # Trace masked average
            obj_masked = (obj * token_mask).sum() / token_mask.sum().clamp(min=1)
            kl = compute_kl_sampled(logp_new, logp_ref, token_mask)

            loss = -(obj_masked - cfg.beta_kl * kl)

            loss_accum = loss_accum + loss
            kl_accum = kl_accum + kl.detach()

        loss_accum = loss_accum / max(1, cfg.group_size)

        accelerator.backward(loss_accum)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        if accelerator.is_main_process and global_step % args.logging_steps == 0:
            logger.info(f'step {global_step} loss {loss_accum.item():.4f} mean_reward {r_t.mean().item():.3f} mean_kl {(kl_accum/cfg.group_size).item() if cfg.group_size>0 else 0.0:.4f}')

        if accelerator.is_main_process and cfg.save_every > 0 and global_step % cfg.save_every == 0:
            out_dir = os.path.join(args.output_dir, f'checkpoint_{global_step}')
            os.makedirs(out_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            logger.info(f'saved checkpoint to {out_dir}')

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info('training finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--solver_model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log_file', type=str, default=None)

    parser.add_argument('--bf16', action='store_true')

    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=900)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--beta_kl', type=float, default=0.02)

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=2000)

    parser.add_argument('--solver_max_new_tokens', type=int, default=256)
    parser.add_argument('--answer_verify', type=str, default='string', choices=['string', 'float', 'auto'])

    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=200)

    args = parser.parse_args()
    main(args)
