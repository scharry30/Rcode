import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

TRACE_START = '<trace>'
TRACE_END = '</trace>'
INSPECT_START = '<inspect>'
INSPECT_END = '</inspect>'


@dataclass
class TraceOp:
    pid: int
    op: str
    arg: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {'id': int(self.pid), 'op': str(self.op).lower()}
        if self.arg is not None:
            out['arg'] = self.arg
        return out


class TraceParseError(ValueError):
    pass


def extract_segment(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.compile(re.escape(start_tag) + r'(.*?)' + re.escape(end_tag), re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def parse_trace_segment(seg: str) -> List[TraceOp]:
    """Parse a trace segment.

    Expected format is a JSON array of dicts, for example:
    [{"id": 1, "op": "keep"}, {"id": 2, "op": "delete"}]

    Supported ops: keep, delete, rewrite, reorder

    For rewrite: arg is the rewritten string
    For reorder: arg can be either an int new position (0 based) or a list of ids specifying full order
    """
    try:
        data = json.loads(seg)
    except Exception as e:
        raise TraceParseError(f"Trace is not valid JSON: {e}")

    if not isinstance(data, list):
        raise TraceParseError('Trace JSON must be a list')

    ops: List[TraceOp] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TraceParseError(f"Trace item {i} must be a dict")
        if 'id' not in item or 'op' not in item:
            raise TraceParseError(f"Trace item {i} must have id and op")
        pid = int(item['id'])
        op = str(item['op']).lower().strip()
        arg = item.get('arg', None)
        if op not in {'keep', 'delete', 'rewrite', 'reorder'}:
            raise TraceParseError(f"Unsupported op {op} at item {i}")
        ops.append(TraceOp(pid=pid, op=op, arg=arg))
    return ops


def parse_inspect_segment(seg: str) -> List[str]:
    """Parse inspected premises.

    Supported formats:

    1) JSON list of strings
    2) Plaintext list where each line is a premise, optionally prefixed by an index like "1." or "(1)"
    """
    seg = seg.strip()
    if not seg:
        return []

    if seg.lstrip().startswith('['):
        try:
            data = json.loads(seg)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [x.strip() for x in data if x.strip()]
        except Exception:
            pass

    lines = [ln.strip() for ln in seg.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r'^\(?\d+\)?\s*[\.:\)]\s*', '', ln)
        ln = re.sub(r'^\d+\s*[\.:\)]\s*', '', ln)
        cleaned.append(ln.strip())
    return [x for x in cleaned if x]


def parse_rts_output(text: str) -> Tuple[List[TraceOp], List[str]]:
    trace_seg = extract_segment(text, TRACE_START, TRACE_END)
    insp_seg = extract_segment(text, INSPECT_START, INSPECT_END)
    if trace_seg is None:
        raise TraceParseError('Missing <trace> segment')
    if insp_seg is None:
        raise TraceParseError('Missing <inspect> segment')
    ops = parse_trace_segment(trace_seg)
    inspected = parse_inspect_segment(insp_seg)
    return ops, inspected


def normalize_premise(p: str) -> str:
    return re.sub(r'\s+', ' ', p.strip().lower())


def execute_trace(premises: List[str], ops: List[TraceOp]) -> List[str]:
    """Deterministic executor E(P, z).

    This applies keep, delete, rewrite on a per premise basis.
    Reorder can be specified as either:

    1) A full order list of ids in a single reorder op, with pid set to 0
    2) A move op with pid referencing the original premise id and arg as new position

    Premise ids are 1 based.
    """

    n = len(premises)
    id_to_text: Dict[int, str] = {i + 1: premises[i] for i in range(n)}
    deleted = set()

    # Apply keep, delete, rewrite first
    for op in ops:
        if op.op == 'reorder':
            continue
        if op.pid < 1 or op.pid > n:
            raise TraceParseError(f"Trace op refers to invalid premise id {op.pid}")
        if op.op == 'delete':
            deleted.add(op.pid)
        elif op.op == 'rewrite':
            if not isinstance(op.arg, str):
                raise TraceParseError('rewrite op arg must be a string')
            id_to_text[op.pid] = op.arg
        elif op.op == 'keep':
            pass

    kept_ids = [pid for pid in range(1, n + 1) if pid not in deleted]

    # Apply reorder
    reorder_ops = [op for op in ops if op.op == 'reorder']
    if reorder_ops:
        # If any op provides full order list, use the last one
        full_orders = [op for op in reorder_ops if isinstance(op.arg, list)]
        if full_orders:
            order = full_orders[-1].arg
            if not all(isinstance(x, int) for x in order):
                raise TraceParseError('reorder full list must be list[int]')
            order = [int(x) for x in order]
            # Filter deleted and unknown ids
            order = [pid for pid in order if pid in id_to_text and pid not in deleted]
            # Append any missing kept ids preserving original order
            for pid in kept_ids:
                if pid not in order:
                    order.append(pid)
            kept_ids = order
        else:
            # Apply move operations in sequence
            for op in reorder_ops:
                if op.pid < 1 or op.pid > n:
                    raise TraceParseError(f"reorder move refers to invalid premise id {op.pid}")
                if op.pid in deleted:
                    continue
                if not isinstance(op.arg, int):
                    raise TraceParseError('reorder move arg must be int position')
                pos = int(op.arg)
                # Remove then insert
                if op.pid in kept_ids:
                    kept_ids.remove(op.pid)
                pos = max(0, min(pos, len(kept_ids)))
                kept_ids.insert(pos, op.pid)

    return [id_to_text[pid].strip() for pid in kept_ids if id_to_text[pid].strip()]


def trace_output_consistent(premises: List[str], ops: List[TraceOp], inspected: List[str]) -> bool:
    exec_out = execute_trace(premises, ops)
    if len(exec_out) != len(inspected):
        return False
    for a, b in zip(exec_out, inspected):
        if normalize_premise(a) != normalize_premise(b):
            return False
    return True
