"""
ingestor.py — Document Parsing Layer  (Week 8 Day 2 — AI Knowledge Assistant)
===============================================================================
Handles extraction of plain text from uploaded files.
Supports: PDF (.pdf), plain text (.txt).

NEW in Week 8 Day 2:
  - extract_many(files) — batch extraction that returns per-file
    Union[IngestPayload, IngestorError] results for bulk upload handling
  - PDF page-level metadata extraction for richer chunking signals
  - Encoding auto-detection with 3 fallbacks for .txt files

Returns a structured IngestPayload dict so the API and vectorstore layers
have no direct dependency on file-format libraries.

Validations enforced here (per file):
  - File extension allowlist
  - Max file size (bytes)
  - Zero-byte file check
  - Non-empty text after extraction
  - Corrupted / unreadable PDF handling
"""

import io
from typing import TypedDict, Union, List, Tuple

import pypdf
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from logger import get_logger

log = get_logger("ingestor")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class IngestPayload(TypedDict):
    text: str
    page_count: int        # Number of pages (PDF) or 1 for txt
    file_type: str         # "pdf" | "txt"
    char_count: int
    filename: str          # Original filename — carried through for tracing


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------
class IngestorError(ValueError):
    """Raised for any file-format or content problem during extraction."""
    def __init__(self, message: str, code: str, filename: str = ""):
        super().__init__(message)
        self.code = code
        self.filename = filename


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_extension(filename: str) -> str:
    """Return lowercase extension or raise IngestorError."""
    name = filename.lower()
    for ext in ALLOWED_EXTENSIONS:
        if name.endswith(ext):
            return ext
    raise IngestorError(
        f"Unsupported file type for '{filename}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        code="INVALID_FILE_TYPE",
        filename=filename,
    )


def _validate_size(data: bytes, filename: str = "") -> None:
    """Raise IngestorError if file is too large or empty."""
    if len(data) == 0:
        raise IngestorError(
            f"File '{filename}' is empty (0 bytes).",
            code="EMPTY_FILE",
            filename=filename,
        )
    if len(data) > MAX_FILE_SIZE_BYTES:
        mb = len(data) / (1024 * 1024)
        limit = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise IngestorError(
            f"File '{filename}' is {mb:.1f} MB — exceeds the {limit:.0f} MB limit.",
            code="FILE_TOO_LARGE",
            filename=filename,
        )


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------
def _extract_pdf(data: bytes, filename: str) -> Tuple[str, int]:
    """
    Return (text, page_count) from PDF bytes.
    Raises IngestorError on corrupted PDFs or scanned-only documents.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception as e:
        raise IngestorError(
            f"Failed to parse PDF '{filename}': {e}",
            code="PARSE_ERROR",
            filename=filename,
        )

    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
        except Exception:
            # Skip unreadable pages — don't fail the whole document
            continue

    if not pages:
        raise IngestorError(
            f"No readable text found in '{filename}'. "
            "Scanned PDFs without OCR are not supported.",
            code="EMPTY_CONTENT",
            filename=filename,
        )

    return "\n".join(pages), len(reader.pages)


def _extract_txt(data: bytes, filename: str) -> Tuple[str, int]:
    """Return (text, 1) from raw text bytes with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(encoding), 1
        except UnicodeDecodeError:
            continue
    raise IngestorError(
        f"Unable to decode '{filename}' — unknown encoding.",
        code="ENCODING_ERROR",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Public API — single file
# ---------------------------------------------------------------------------
def extract(filename: str, data: bytes) -> IngestPayload:
    """
    Entry point for single-document extraction.

    Args:
        filename: Original filename (used to detect format and for logging).
        data:     Raw file bytes.

    Returns:
        IngestPayload with text, page_count, file_type, char_count, filename.

    Raises:
        IngestorError: On any validation or extraction failure.
    """
    # 1. Extension check
    ext = _validate_extension(filename)

    # 2. Size and empty check
    _validate_size(data, filename)

    log.info("Extracting document", extra={"filename": filename, "bytes": len(data)})

    # 3. Format dispatch
    if ext == ".pdf":
        text, pages = _extract_pdf(data, filename)
        file_type = "pdf"
    elif ext == ".txt":
        text, pages = _extract_txt(data, filename)
        file_type = "txt"
    else:
        # Guard — should never reach here after _validate_extension
        raise IngestorError("Unhandled extension.", code="INVALID_FILE_TYPE", filename=filename)

    # 4. Non-empty content check
    text = text.strip()
    if not text:
        raise IngestorError(
            f"No readable text found in '{filename}' after extraction.",
            code="EMPTY_CONTENT",
            filename=filename,
        )

    log.info(
        "Extraction complete",
        extra={
            "filename": filename,
            "file_type": file_type,
            "char_count": len(text),
            "pages": pages,
        },
    )

    return IngestPayload(
        text=text,
        page_count=pages,
        file_type=file_type,
        char_count=len(text),
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Public API — batch  (NEW Week 8 Day 2)
# ---------------------------------------------------------------------------
def extract_many(
    files: List[Tuple[str, bytes]]
) -> List[Union[IngestPayload, IngestorError]]:
    """
    Extract text from multiple files sequentially.

    Processes each file independently — a failure in one does NOT abort the
    remaining files. The caller is responsible for checking result types.

    Args:
        files: List of (filename, raw_bytes) tuples, in order.

    Returns:
        List of the same length as `files`. Each element is either:
          - IngestPayload  → extraction succeeded
          - IngestorError  → extraction failed for that file
    """
    results: List[Union[IngestPayload, IngestorError]] = []
    for filename, data in files:
        try:
            payload = extract(filename, data)
            results.append(payload)
        except IngestorError as e:
            log.warning(
                "Batch extraction failed for file",
                extra={"filename": filename, "code": e.code, "error": str(e)},
            )
            results.append(e)
        except Exception as e:
            # Unexpected error — wrap it
            err = IngestorError(
                f"Unexpected error extracting '{filename}': {e}",
                code="EXTRACTION_ERROR",
                filename=filename,
            )
            log.error(
                "Unexpected extraction error",
                extra={"filename": filename, "error": str(e)},
            )
            results.append(err)
    return results
