"""Mask recognizable contact details before research CVs enter the model/index.

This is a deterministic contact filter, not a guarantee of full anonymization.
Names, employers, locations and obfuscated contact details can remain.
The implementation lives in ``src.backend.text_cleaning`` so the ingest graph
applies the same filter to uploaded PDFs.
"""

from src.backend.text_cleaning import REDACTION_VERSION, redact_contacts

__all__ = ["REDACTION_VERSION", "redact_contacts"]
