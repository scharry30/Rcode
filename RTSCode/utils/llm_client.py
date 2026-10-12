from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def chat(self, messages: Sequence[ChatMessage], max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Optional remote client.

    This is provided for convenience. It requires openai>=1.0.0 and an API key in environment.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError('OpenAIClient requires openai package. Install openai>=1.0.0') from e

        if api_key is None:
            import os
            api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY is required for OpenAIClient')

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def chat(self, messages: Sequence[ChatMessage], max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0) -> str:
        payload = [{'role': m.role, 'content': m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content


class HFLocalClient(LLMClient):
    """Local HuggingFace generation client."""

    def __init__(self, model_name_or_path: str, device_map: str = 'auto', torch_dtype: str = 'auto'):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(self, messages: Sequence[ChatMessage], max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0) -> str:
        # Basic chat template. For best results, pass a model with a proper chat template.
        prompt = ''
        for m in messages:
            prompt += f"{m.role.upper()}: {m.content}\n"
        prompt += 'ASSISTANT: '

        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        out = self.tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return out.strip()
