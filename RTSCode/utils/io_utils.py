import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_json(path: str, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        raise RuntimeError(f"Failed to read json from {path}: {e}")


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse jsonl at {path}:{line_no}: {e}")


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(iter_jsonl(path)):
        out.append(item)
        if limit is not None and i + 1 >= limit:
            break
    return out


def write_jsonl(path: str, items: Iterable[Dict[str, Any]], append: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write('\n')


def shard_iter(items: List[Any], shard_id: int, num_shards: int) -> List[Any]:
    if num_shards <= 1:
        return items
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id {shard_id} for num_shards {num_shards}")
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]
