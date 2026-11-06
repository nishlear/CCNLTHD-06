# Benchmark dataset

Small evaluation set for the CV/JD matching app. Built from the Kaggle resume dataset at `data/archive/Resume/Resume.csv`.

## Files

| File | Rows | What it is |
|---|---|---|
| `cvs.jsonl` | 100 | Sampled CVs. `{"cv_id", "category", "text"}`. `text` is `Resume_str` with whitespace collapsed. |
| `jds.jsonl` | 15 | Job descriptions. `{"jd_id", "category", "text", "gold": {"skills", "min_years", "level"}}`. |
| `gold_extract.jsonl` | 30 | Extraction labels for 30 CVs. `{"cv_id", "skills", "years_experience", "labeled_by"}`. |
| `make_sample.py` | - | Re-creates `cvs.jsonl`. Run from repo root: `python data/benchmark/make_sample.py`. |

## Sampling

- 10 categories: INFORMATION-TECHNOLOGY, ENGINEERING, FINANCE, ACCOUNTANT, BANKING, CHEF, HEALTHCARE, DESIGNER, DIGITAL-MEDIA, SALES.
- Picked to mix easy categories (CHEF, HEALTHCARE, SALES) with groups that are easy to confuse:
  FINANCE / ACCOUNTANT / BANKING, DESIGNER / DIGITAL-MEDIA, INFORMATION-TECHNOLOGY / ENGINEERING.
- 10 CVs per category, `random.Random(42).sample` over each category's pool (sorted by ID).
- CVs with `Resume_str` under 500 characters are skipped (none in these 10 categories were that short).

## JDs

**All 15 JDs are synthetic**, written by Claude. One per category (jd01-jd10), plus 5 harder ones in the confusable groups (jd11 FINANCE, jd12 BANKING, jd13 DESIGNER, jd14 DIGITAL-MEDIA, jd15 ENGINEERING). `gold.skills` are copied as written in the JD text. `gold.min_years` and `gold.level` are stated in the text.

## gold_extract labels

**These are Claude drafts (`labeled_by: "claude-draft-v2"`). The team must review them before any number is reported.**

- 30 CVs: the first 3 of each category in `cvs.jsonl` order.
- `skills`: skills written in the CV text, deduplicated, kept as written. Not inferred.
- `years_experience` (v2 rule, labeled by reading each CV's work history, no regex):
  - Count all dated roles: full-time, part-time, student jobs, internships, research positions, military service.
  - Month granularity, both ends inclusive: "01/2010 to 06/2010" = 6 months. Year-only ranges ("2010 - 2012") run Jan of start year to Dec of end year. A role with a single year counts 12 months.
  - Overlapping roles count once: union of month ranges across all roles, then total months / 12, rounded to 1 decimal.
  - "Current"/"Present" ends at December of the latest year seen anywhere in the CV (the CVs are old). Future-dated items like "valid through 2017" on a certificate are ignored when finding that year.
  - Roles with no dates are left out. Reversed ranges (09/2008 to 04/2005) are read as 04/2005-09/2008.
  - `null` only if the CV has no usable dates at all.
