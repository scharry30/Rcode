from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.io_utils import read_jsonl
from utils.prompts import build_inspection_user_prompt, DEFAULT_INSPECT_SYSTEM
from utils.logging_utils import get_logger


def build_sft_example(question: str, premises: List[str], trace: List[Dict], inspected_premises: List[str]) -> Dict[str, str]:
    user_prompt = build_inspection_user_prompt(question, premises)
    target = '<trace>' + json.dumps(trace, ensure_ascii=False) + '</trace>\n' + '<inspect>' + '\n'.join([f'{i+1}. {p}' for i, p in enumerate(inspected_premises)]) + '</inspect>'
    full_prompt = f'SYSTEM: {DEFAULT_INSPECT_SYSTEM}\nUSER: {user_prompt}\nASSISTANT: '
    return {'prompt': full_prompt, 'completion': target}


class CompletionOnlyCollator:
    """Data collator that masks loss on the prompt part.

    It searches for the separator token sequence representing the start of the completion.
    """

    def __init__(self, tokenizer: AutoTokenizer, completion_start: str = 'ASSISTANT: '):
        self.tokenizer = tokenizer
        self.completion_start = completion_start

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [f['prompt'] + f['completion'] for f in features]
        enc = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

        labels = enc['input_ids'].clone()
        for i, f in enumerate(features):
            prefix = f['prompt']
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)['input_ids']
            labels[i, :len(prefix_ids)] = -100
        enc['labels'] = labels
        return enc


def main(args):
    logger = get_logger('rts_sft', log_file=args.log_file)

    raw = read_jsonl(args.train_jsonl, limit=args.limit)
    logger.info(f'Loaded training rows: {len(raw)}')

    examples: List[Dict[str, str]] = []
    for row in raw:
        examples.append(build_sft_example(row['question'], row['premises'], row['trace'], row['inspected_premises']))

    ds = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model_kwargs['quantization_config'] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16 if args.bf16 else None, device_map='auto', **model_kwargs)

    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=args.lora_target_modules.split(',') if args.lora_target_modules else None,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    collator = CompletionOnlyCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log_file', type=str, default=None)

    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')

    args = parser.parse_args()
    main(args)
