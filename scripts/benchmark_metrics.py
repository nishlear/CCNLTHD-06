"""Pure helpers for scripts/benchmark.py: metrics, cost, error types and the call cache.

No network and no app config imports here, so everything is unit-testable.
"""

import hashlib
import json
import math
import re
from pathlib import Path


# ---- Skills ----

def norm_skill(skill) -> str:
    """Case-insensitive, whitespace-normalized form used for exact matching."""
    return " ".join(str(skill).casefold().split())


def _skill_set(skills) -> set[str]:
    return {s for s in (norm_skill(x) for x in (skills or []) if x is not None) if s}


def _contains(a: str, b: str) -> bool:
    """True when one string contains the other on word boundaries.

    Boundaries stop 1-letter skills from matching everything ("c" in "css",
    "java" in "javascript") while "react" still matches "react.js".
    """
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", long_) is not None


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def skill_prf(predicted, gold, lenient: bool = False) -> tuple[float, float, float]:
    """Precision, recall, F1 of one item's skill list.

    Both empty -> (1, 1, 1). Empty prediction with non-empty gold -> (0, 0, 0).
    Lenient: a predicted skill counts as correct when it contains, or is contained
    in, any gold skill (and the same for recall in the other direction).
    """
    pred, ref = _skill_set(predicted), _skill_set(gold)
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    if lenient:
        p = sum(any(_contains(x, g) for g in ref) for x in pred) / len(pred)
        r = sum(any(_contains(g, x) for x in pred) for g in ref) / len(ref)
    else:
        hit = len(pred & ref)
        p, r = hit / len(pred), hit / len(ref)
    return p, r, _f1(p, r)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def mean_prf(rows) -> tuple:
    """Macro average of (p, r, f1) tuples; F1 is averaged per item, not recomputed."""
    rows = list(rows)
    if not rows:
        return None, None, None
    return tuple(mean([row[i] for row in rows]) for i in range(3))


# ---- Years ----

def years_metrics(pairs) -> dict:
    """pairs: (predicted, gold). Only pairs where both are known count for MAE.

    Returns mae, within_1 (share with |error| <= 1), n (scored pairs) and
    missing_rate (gold known but prediction null).
    """
    known_gold = [(p, g) for p, g in pairs if g is not None]
    scored = [(p, g) for p, g in known_gold if p is not None]
    errors = [abs(float(p) - float(g)) for p, g in scored]
    return {"mae": mean(errors),
            "within_1": (sum(e <= 1 for e in errors) / len(errors)) if errors else None,
            "n": len(errors),
            "missing_rate": ((len(known_gold) - len(scored)) / len(known_gold)) if known_gold else None}


def same_number(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 1e-9


# ---- Ranking ----

def precision_at_k(relevant: list[bool], k: int) -> float:
    """Share of the top-k that is relevant; a list shorter than k counts missing slots as misses."""
    return sum(bool(x) for x in relevant[:k]) / k


def reciprocal_rank(relevant: list[bool]) -> float:
    for i, hit in enumerate(relevant, 1):
        if hit:
            return 1.0 / i
    return 0.0


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def rank_by_cosine(query, docs: dict) -> list[tuple[str, float]]:
    """Rank doc ids by cosine. Vectors are already L2-normalized, so cosine = dot product.

    Ties break by id so results are deterministic. No similarity threshold.
    """
    scored = [(doc_id, dot(query, vec)) for doc_id, vec in docs.items()]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def percentile(values, q: float):
    """Linear-interpolated percentile (q in 0..100), like numpy's default."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pos = (len(values) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


# ---- Cost ----

def chat_cost(input_tokens, output_tokens, price):
    """USD for one chat call; price = (in, out) per 1M tokens. None when unknown."""
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def embed_cost(tokens, price):
    """USD for embedding `tokens` input tokens; price per 1M tokens. None when unknown."""
    if price is None or tokens is None:
        return None
    return tokens * price / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token count (chars / 4) when the API reports no usage."""
    return max(1, round(len(text) / 4))


def split_tokens(total: int, texts: list[str]) -> list[int]:
    """Split a batch's reported token count over its texts in proportion to length."""
    weights = [max(1, len(t)) for t in texts]
    shares = [total * w / sum(weights) for w in weights]
    out = [math.floor(s) for s in shares]
    for i in sorted(range(len(out)), key=lambda i: out[i] - shares[i])[:total - sum(out)]:
        out[i] += 1
    return out


def sum_known(values):
    """Sum of non-None values; None if every value is None (or there are none)."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


# ---- Errors ----

def classify_error(exc: BaseException) -> str:
    """Map an exception to a benchmark error type.

    Checked by class name so this module does not import the app (which reads config at import).
    pydantic.ValidationError and json.JSONDecodeError both subclass ValueError, so order matters.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "ValidationError" in names:
        return "schema_invalid"
    if "JSONDecodeError" in names or (isinstance(exc, ValueError) and "not a JSON object" in str(exc)):
        return "json_invalid"
    if "GoogleRateLimitError" in names:
        return "rate_limit"
    if names & {"GeminiTimeoutError", "TimeoutError", "TimeoutException"}:
        return "timeout"
    if "ConfigurationError" in names:
        return "config"
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 429:
            return "rate_limit"
        if code in (408, 504):
            return "timeout"
        return f"http_{code}"
    if names & {"URLError", "ConnectionError", "ConnectError"}:
        return "connection"
    return "other_" + type(exc).__name__


# ---- Cache ----

def input_sha(value) -> str:
    """sha256 of the exact model input (canonical JSON for dicts)."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_key(task: str, model: str, sha: str) -> str:
    return f"{task}|{model}|{sha}"


class CallCache:
    """Append-only JSONL cache of model calls. The last record for a key wins."""

    def __init__(self, path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from an interrupted run
                    self.records[record["key"]] = record

    def get(self, task: str, model: str, sha: str) -> dict | None:
        return self.records.get(cache_key(task, model, sha))

    def append(self, record: dict) -> None:
        record = {**record, "key": cache_key(record["task"], record["model"], record["input_sha"])}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[record["key"]] = record

    def pending(self, task: str, model: str, items: list[dict], retry_errors: bool = False) -> list[dict]:
        """Items (dicts with 'input') that still need a call; duplicates by input are dropped."""
        out, seen = [], set()
        for item in items:
            sha = input_sha(item["input"])
            if sha in seen:
                continue
            seen.add(sha)
            record = self.get(task, model, sha)
            if record is None or (retry_errors and not record.get("ok")):
                out.append(item)
        return out
