import re
from dataclasses import dataclass
from typing import Optional, Tuple


def extract_final_answer(text: str) -> str:
    if text is None:
        return ''
    t = text.strip()
    # common patterns
    m = re.search(r'final\s*answer\s*[:\-]?\s*(.*)$', t, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    # take last non empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else t


def normalize_number(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s', '', text)
    sci = re.search(r'([-+]?\d*\.?\d+)[×xX]10\^?([-+]?\d+)', text)
    if sci:
        base = float(sci.group(1))
        exp = int(sci.group(2))
        return str(base * (10 ** exp))
    m = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
    if m:
        return m.group(1)
    return text


def verify_float(reference: str, prediction: str, tolerance: float = 0.05) -> bool:
    try:
        ref_num = float(normalize_number(reference))
        pred_num = float(normalize_number(prediction))
        if ref_num == 0:
            return abs(pred_num) < tolerance
        rel = abs((pred_num - ref_num) / ref_num)
        return rel <= tolerance
    except Exception:
        return normalize_number(reference) == normalize_number(prediction)


def verify_string(reference: str, prediction: str) -> bool:
    ref_clean = re.sub(r'\s', '', str(reference).lower())
    pred_clean = re.sub(r'\s', '', str(prediction).lower())
    return ref_clean == pred_clean


def verify_answer(reference: str, prediction: str, method: str = 'float', tolerance: float = 0.05) -> bool:
    if method == 'float':
        return verify_float(reference, prediction, tolerance)
    if method == 'string':
        return verify_string(reference, prediction)
    try:
        return verify_float(reference, prediction, tolerance)
    except Exception:
        return verify_string(reference, prediction)


@dataclass
class AccuracyResult:
    correct: bool
    pred: str
    gold: str


def score_prediction(gold: str, model_output: str, method: str = 'float', tolerance: float = 0.05) -> AccuracyResult:
    pred = extract_final_answer(model_output)
    ok = verify_answer(gold, pred, method=method, tolerance=tolerance)
    return AccuracyResult(correct=ok, pred=pred, gold=str(gold))
