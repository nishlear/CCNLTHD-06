"""Create a reproducible sample of real, unique Djinni CV texts; no model calls."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

DATASET = "lang-uk/recruitment-dataset-candidate-profiles-english"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
SOURCE_REVISION = "f4d04f624126e27a17770d06953dce7e914ce214"
EXPECTED_SHA256 = "a7b1378e0b231507c377659bf2c80fa4ba60c4e31e1107b13425d103879ecc3b"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_records(rows, limit, seed, min_chars=80, max_chars=24000):
    rng = random.Random(seed)
    sample, seen_ids, seen_text = [], set(), set()
    counts = Counter()
    for row in rows:
        counts["source_rows"] += 1
        text = str(row.get("CV") or "").strip()
        source_id = str(row.get("id") or "").strip()
        position = str(row.get("Position") or "").strip()
        if not source_id or not text or row.get("CV_lang") != "en":
            counts["missing_id_text_or_non_english"] += 1
            continue
        if not min_chars <= len(text) <= max_chars:
            counts["invalid_text_length"] += 1
            continue
        fingerprint = hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()
        if source_id in seen_ids or fingerprint in seen_text:
            counts["duplicate_id_or_cv_text"] += 1
            continue
        seen_ids.add(source_id)
        seen_text.add(fingerprint)
        counts["eligible_unique"] += 1
        years = row.get("Experience Years")
        years = float(years) if isinstance(years, (int, float)) and math.isfinite(years) else None
        item = {
            "cv_id": "djinni-" + source_id,
            "source_id": source_id,
            "source_dataset": DATASET,
            "source_row": counts["source_rows"] - 1,
            "text_sha256": fingerprint,
            "display_name": "Anonymous candidate " + source_id[:8],
            "file_name": "djinni-" + source_id + ".txt",
            "raw_text": ("Position: " + position + "\n\n" + text).strip(),
            "position": position,
            "source_primary_keyword": row.get("Primary Keyword"),
            # Keep source metadata separate from the model-extracted experience.
            "source_experience_years": years,
        }
        if len(sample) < limit:
            sample.append(item)
        else:
            slot = rng.randrange(counts["eligible_unique"])
            if slot < limit:
                sample[slot] = item
    return sample, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/datasets/djinni/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/datasets/djinni/sample-100000.jsonl"))
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    if args.output.exists():
        parser.error("output already exists; choose another output path")
    import pyarrow.parquet as pq
    started = perf_counter()
    digest = sha256_file(args.source)
    if digest != EXPECTED_SHA256:
        raise ValueError("Dataset SHA256 differs from the verified source")
    parquet = pq.ParquetFile(args.source)
    rows = (row for batch in parquet.iter_batches(batch_size=1024) for row in batch.to_pylist())
    sample, counts = sample_records(rows, args.limit, args.seed)
    if len(sample) != args.limit:
        raise ValueError(f"Only {len(sample)} eligible unique CVs; requested {args.limit}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with partial.open("wb") as handle:
        for item in sample:
            handle.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
    partial.replace(args.output)
    lengths = sorted(len(item["raw_text"]) for item in sample)
    manifest = {
        "dataset": DATASET, "dataset_url": DATASET_URL, "dataset_license": "MIT",
        "source_revision": SOURCE_REVISION, "source_sha256": digest,
        "source_file_bytes": args.source.stat().st_size, "seed": args.seed,
        "sampling": "reservoir sample of eligible unique English CV texts",
        "duplicate_definition": "same source ID or SHA256 of casefolded whitespace-normalized CV text",
        **counts, "selected": len(sample), "output_file": str(args.output),
        "output_bytes": args.output.stat().st_size, "output_sha256": sha256_file(args.output),
        "text_characters": {"min": lengths[0], "median": lengths[len(lengths)//2],
                            "max": lengths[-1], "total": sum(lengths)},
        "source_categories": dict(Counter(item["source_primary_keyword"] or "unknown" for item in sample)),
        "elapsed_seconds": round(perf_counter() - started, 3), "openai_requests": 0,
        "note": "Real candidate-profile text, not original PDF files. Display names are anonymous IDs. "
                "Source experience metadata is not asserted to be exact years. No extraction, embedding "
                "or matching-quality evaluation is claimed by this preparation step.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_categories"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
