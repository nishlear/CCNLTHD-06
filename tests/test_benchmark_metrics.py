import json
import urllib.error

import pytest
from pydantic import BaseModel

from scripts import benchmark_metrics as bm


def test_skill_f1_normalizes_case_and_whitespace():
    p, r, f = bm.skill_prf(["  Python ", "machine   learning", "SQL"], ["python", "Machine Learning", "Go"])
    assert (p, r) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert f == pytest.approx(2 / 3)


def test_skill_f1_edge_cases():
    assert bm.skill_prf([], []) == (1.0, 1.0, 1.0)
    assert bm.skill_prf(None, ["Python"]) == (0.0, 0.0, 0.0)
    assert bm.skill_prf(["Python"], []) == (0.0, 0.0, 0.0)
    # duplicates after normalization count once
    assert bm.skill_prf(["Python", "python "], ["Python"]) == (1.0, 1.0, 1.0)


def test_lenient_match_uses_containment_on_word_boundaries():
    assert bm.skill_prf(["React.js"], ["react"], lenient=True) == (1.0, 1.0, 1.0)
    assert bm.skill_prf(["React.js"], ["react"])[2] == 0.0
    # no partial-word matches
    assert bm.skill_prf(["JavaScript"], ["Java"], lenient=True)[2] == 0.0
    assert bm.skill_prf(["CSS"], ["C"], lenient=True)[2] == 0.0
    p, r, _ = bm.skill_prf(["Microsoft Excel", "Word"], ["Excel"], lenient=True)
    assert (p, r) == (0.5, 1.0)


def test_mean_prf_is_macro_average():
    assert bm.mean_prf([(1, 1, 1), (0, 0, 0)]) == (0.5, 0.5, 0.5)
    assert bm.mean_prf([]) == (None, None, None)


def test_years_mae_within_one_and_missing():
    out = bm.years_metrics([(5, 4), (2, 5), (None, 3), (7, None)])
    assert out["mae"] == pytest.approx(2.0)
    assert out["within_1"] == 0.5
    assert out["n"] == 2
    assert out["missing_rate"] == pytest.approx(1 / 3)
    assert bm.years_metrics([])["mae"] is None


def test_same_number_treats_two_nulls_as_equal():
    assert bm.same_number(None, None)
    assert bm.same_number(3, 3.0)
    assert not bm.same_number(None, 0)
    assert not bm.same_number(2, 3)


def test_precision_at_k_counts_missing_slots_as_misses():
    rel = [True, False, True, True, False, False]
    assert bm.precision_at_k(rel, 5) == 0.6
    assert bm.precision_at_k(rel, 10) == 0.3
    assert bm.precision_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert bm.reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert bm.reciprocal_rank([True]) == 1.0
    assert bm.reciprocal_rank([False, False]) == 0.0


def test_rank_by_cosine_has_no_threshold_and_breaks_ties_by_id():
    docs = {"b": [1.0, 0.0], "a": [1.0, 0.0], "c": [0.0, 1.0], "d": [-1.0, 0.0]}
    ranked = bm.rank_by_cosine([1.0, 0.0], docs)
    assert [doc for doc, _ in ranked] == ["a", "b", "c", "d"]
    assert ranked[-1][1] == -1.0


def test_percentile_interpolates():
    assert bm.percentile([1, 2, 3, 4], 50) == 2.5
    assert bm.percentile([10], 95) == 10
    assert bm.percentile([None], 50) is None
    assert bm.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_cost_with_known_and_none_price():
    assert bm.chat_cost(1_000_000, 2_000_000, (0.30, 2.50)) == pytest.approx(5.30)
    assert bm.chat_cost(100, 10, None) is None
    assert bm.chat_cost(None, 10, (0.1, 0.5)) is None
    assert bm.chat_cost(500, 50, (0.0, 0.0)) == 0.0
    assert bm.embed_cost(1_000_000, 0.02) == pytest.approx(0.02)
    assert bm.embed_cost(1000, None) is None
    assert bm.sum_known([None, None]) is None
    assert bm.sum_known([None, 1.5, 2]) == 3.5


def test_token_estimate_and_split():
    assert bm.estimate_tokens("a" * 40) == 10
    assert bm.estimate_tokens("") == 1
    parts = bm.split_tokens(10, ["aaa", "a", "aaaaaa"])
    assert sum(parts) == 10 and parts[2] >= parts[0] >= parts[1]


def test_error_classification():
    class Strict(BaseModel):
        x: int
    try:
        Strict.model_validate({"x": "nope"})
    except Exception as exc:
        assert bm.classify_error(exc) == "schema_invalid"
    try:
        json.loads("{bad")
    except Exception as exc:
        assert bm.classify_error(exc) == "json_invalid"
    assert bm.classify_error(ValueError("Model output is not a JSON object")) == "json_invalid"
    GoogleRateLimitError = type("GoogleRateLimitError", (RuntimeError,), {})
    GeminiTimeoutError = type("GeminiTimeoutError", (TimeoutError,), {})
    assert bm.classify_error(GoogleRateLimitError()) == "rate_limit"
    assert bm.classify_error(GeminiTimeoutError()) == "timeout"
    http = lambda code: urllib.error.HTTPError("u", code, "m", {}, None)
    assert bm.classify_error(http(429)) == "rate_limit"
    assert bm.classify_error(http(500)) == "http_500"
    assert bm.classify_error(urllib.error.URLError("refused")) == "connection"
    assert bm.classify_error(KeyError("x")) == "other_KeyError"


def test_input_sha_is_canonical_for_dicts():
    assert bm.input_sha({"a": 1, "b": "é"}) == bm.input_sha({"b": "é", "a": 1})
    assert bm.input_sha("x") != bm.input_sha("y")
    assert bm.cache_key("extract", "m", "abc") == "extract|m|abc"


def test_cache_skips_done_items_and_retries_errors_only_on_request(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = bm.CallCache(path)
    items = [{"id": "1", "input": {"cv": "one"}}, {"id": "2", "input": {"cv": "two"}},
             {"id": "3", "input": {"cv": "three"}}, {"id": "1b", "input": {"cv": "one"}}]
    cache.append({"task": "extract", "model": "m", "input_sha": bm.input_sha(items[0]["input"]),
                  "item_id": "1", "ok": True, "error": None})
    cache.append({"task": "extract", "model": "m", "input_sha": bm.input_sha(items[1]["input"]),
                  "item_id": "2", "ok": False, "error": "json_invalid"})
    reloaded = bm.CallCache(path)  # persisted across runs
    assert [i["id"] for i in reloaded.pending("extract", "m", items)] == ["3"]
    assert [i["id"] for i in reloaded.pending("extract", "m", items, retry_errors=True)] == ["2", "3"]
    # a different model or task is a different key
    assert len(reloaded.pending("extract", "other", items)) == 3
    assert len(reloaded.pending("parse", "m", items)) == 3


def test_cache_last_record_wins_and_torn_lines_are_ignored(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = bm.CallCache(path)
    sha = bm.input_sha("q")
    cache.append({"task": "parse", "model": "m", "input_sha": sha, "ok": False, "error": "timeout"})
    cache.append({"task": "parse", "model": "m", "input_sha": sha, "ok": True, "error": None})
    with path.open("a") as fh:
        fh.write('{"key": "trunc')
    reloaded = bm.CallCache(path)
    assert reloaded.get("parse", "m", sha)["ok"] is True
    assert reloaded.pending("parse", "m", [{"id": "x", "input": "q"}], retry_errors=True) == []
