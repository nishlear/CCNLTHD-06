# Real CV corpus and reproducible runs

Source: [Djinni English candidate profiles](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english).
Paper: Nazarii Drushchak and Mariana Romanyshyn (2024), [Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings](https://aclanthology.org/2024.unlp-1.2/).
The dataset card declares the MIT license. The related source-code repository has its own license; the two are separate.

The downloaded file contains real anonymized candidate-profile **text**, not 210,250 original PDF documents. It is not a synthetic-resume dataset. It supports search/indexing experiments and extraction from text; PDF parsing needs its own PDF evaluation set.

## Verified local data

| Item | Value |
|---|---|
| Source revision | `f4d04f624126e27a17770d06953dce7e914ce214` |
| Source Parquet | `data/datasets/djinni/train.parquet` |
| Source SHA256 | `a7b1378e0b231507c377659bf2c80fa4ba60c4e31e1107b13425d103879ecc3b` |
| Source bytes / rows | 237,415,736 / 210,250 |
| Eligible unique rows | 209,638 |
| Excluded for CV text length | 612 |
| Prepared sample | 100,000 profiles; seed 20261023 |
| Sample JSONL | `data/datasets/djinni/sample-100000.jsonl` |
| Sample SHA256 | `e446c3fa164a3003f453f28ad6130fbda79cd2f31f8b86620edabcf338984f8d` |
| Sample bytes | 153,764,041 |
| Text length, min / median / max | 100 / 786 / 7,406 characters including position |

Sampling scans the whole file and uses reservoir sampling over eligible unique English CV texts. It rejects missing IDs, empty/non-English text and CV texts outside 80–24,000 characters. Duplicates are defined by source ID or SHA256 of casefolded, whitespace-normalized CV text. Display names such as `Anonymous candidate <id>` identify records and do not assert a person's name. Source experience metadata is preserved separately; it is not asserted to be exact years or used as extraction ground truth.

To recreate the sample using installed project dependencies:

```powershell
uv sync --extra research
uv run --extra research python -m scripts.prepare_djinni
```

Use another `--output` path if the prepared sample already exists. Its adjacent manifest records input/output hashes, filtering, category counts and timing. Raw data, keys, dependency downloads and generated benchmarks stay under ignored paths.

## Measured local lexical baseline

The actual 100,000-profile JSONL was indexed with SQLite FTS5, using SQLite 3.45.1. Build time was 5.304 seconds and the database was 189,456,384 bytes. Each top-10 query was repeated 20 times after building the index, so these are warm-cache measurements.

| Keyword query | Matching profiles | p50 ms | p95 ms |
|---|---:|---:|---:|
| Python AND FastAPI | 494 | 0.803 | 1.243 |
| Java AND Spring | 4,035 | 3.028 | 3.315 |
| React AND TypeScript | 4,380 | 4.059 | 6.755 |
| Docker AND Kubernetes | 1,665 | 1.743 | 2.697 |
| SQL AND data | 4,363 | 7.084 | 9.521 |

This is an explicitly separate lexical baseline. These numbers do not measure Chroma ANN, model extraction, OpenAI latency, overall application latency or hiring suitability. No relevance labels were supplied, so precision, recall and nDCG were not measured. The baseline made zero OpenAI/embedding requests.

```powershell
python -m scripts.benchmark_corpus --database data/benchmarks/another-fts.sqlite --report data/benchmarks/another-fts.json
```

## Current Gemini model/index workflow

```powershell
# First verify access and run a 100-profile pilot.
.\scripts\run-real-data.ps1 -Limit 100
# Then continue the same corpus to 100,000, reusing ready records.
.\scripts\run-real-data.ps1 -Limit 100000
```

The script updates dependencies and the lock, runs tests, checks official PyPI/OpenAI endpoints and imports contact-filtered CV text through the actual ingest graph in batches of at most 20. It stops after three entirely failed batches, saves progress and skips already-ready records on restart only when their stored text matches the current filtered text. CV extraction and embeddings are configured for Google Gemini (gemini-3.8-flash and gemini-embedding-2). Search creates a genuine checkpoint awaiting HR review; it does not automatically approve the shortlist. API model use incurs provider charges.

In the editing environment, the `uv 0.13.5` and `PyArrow 26.0.0` downloads succeeded through the browser and hashes were verified. Direct terminal access to PyPI and OpenAI failed with DNS error 11001. The full application dependency update, lock refresh and OpenAI/LangGraph/Chroma run therefore remain **unverified and incomplete**. `data/benchmarks/runtime-check.json` records endpoint results without credentials. The real sample and lexical baseline above completed locally.

## Contact filtering before API calls

The research importer always applies `scripts/redact_contacts.py` before passing any CV text to the ingest graph, OpenAI extraction, embeddings or the application database. The filter masks recognizable emails, web URLs, common bare profile links, labelled messenger handles and phone-like sequences. Its version and aggregate replacement counts are recorded in the run report; matched values are not logged. Source Parquet/JSONL files and the original lexical baseline remain unchanged locally. An older ready row containing unfiltered text is reprocessed rather than reused.

This is a contact filter, not a guarantee of full anonymization. Names, employers, locations, project descriptions and obfuscated contact details can remain identifiable. The filtering described here applies to the research importer.

Tests of contact removal, preservation of programming skills, idempotence and the import/model boundary run without application dependencies:

```powershell
python -m unittest discover -s tests -p test_contact_redaction.py -v
```

A local filter pass over all 100,000 prepared profiles changed 1,277 profiles: 2,811 URL replacements, 16 bare-profile-link replacements, 8 phone-like replacements and 6 labelled messenger replacements. The earlier basic email/profile/phone patterns had zero matches in the resulting text. These are heuristic filter counts, not confirmed counts of personal contacts or a comprehensive privacy audit. No OpenAI requests were made by this check.

## Report update on 2026-10-24

[The recorded experiment chapter](THUC_NGHIEM_2026-10-24.md) contains the actual contact-filtered 100,000-profile run and the incomplete OpenAI pilot. The filtered FTS build took 26.305 seconds including 17.916 seconds of filtering. Five query groups were timed 50 times each. This is a different measurement from the original unfiltered baseline above; the results are kept separate.

```powershell
python -m scripts.benchmark_corpus --redact-contacts --database data/benchmarks/another-filtered.sqlite --report data/benchmarks/another-filtered.json --repeats 50
```

Chapter 12 and its table of contents were updated directly in the linked Google Doc. A separate evidence tab records aggregate results, commands and script hashes. The local JSON summary is `data/benchmarks/experiments_20261024.json`; the attempted pilot processed zero CVs because dependency installation failed at DNS resolution. Neither the report nor the JSON summary claims a successful OpenAI run.

## Gemini provider migration

Current model calls use Gemini, with `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env`. First run `python -m scripts.check_gemini_model`, then `scripts/run-real-data.ps1 -Limit 100`. New run reports use `data/benchmarks/gemini-dataset-run.json` and record provider/model/embedding settings. Switching providers requires a new vector collection; an old ready database row is not reused unless its current-collection vector exists. The contact filter still runs before any CV is sent to Gemini.

The OpenAI/DNS results above describe the historical attempt and are retained as measured. They are not evidence of Gemini usage. Gemini inference, extraction and embedding remain unverified until a valid Google key and successful provider responses are recorded.
