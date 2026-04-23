"""
ingestor.py — Document Parsing Layer
======================================
Handles extraction of plain text from uploaded files.
Supports: PDF (.pdf), plain text (.txt), Word documents (.docx).

Returns a structured IngestPayload dict so the API and vectorstore layers
have no direct dependency on file-format libraries.

Validations enforced here:
  - File extension allowlist
  - Max file size (bytes)
  - Non-empty text after extraction
  - UTF-8 encoding fallback for .txt files
"""

import io
from typing import TypedDict

import pypdf
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from logger import get_logger

log = get_logger("ingestor")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class IngestPayload(TypedDict):
    text: str
    page_count: int      # Number of pages/sections
    file_type: str       # "pdf" | "txt" | "docx"
    char_count: int


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class IngestorError(ValueError):
    """Raised for any file-format or content problem during extraction."""
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def _validate_extension(filename: str) -> str:
    """Return lowercase extension or raise IngestorError."""
    name = filename.lower()
    for ext in ALLOWED_EXTENSIONS:
        if name.endswith(ext):
            return ext
    raise IngestorError(
        f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        code="INVALID_FILE_TYPE",
    )


def _validate_size(data: bytes) -> None:
    if len(data) > MAX_FILE_SIZE_BYTES:
        mb = len(data) / (1024 * 1024)
        limit = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise IngestorError(
            f"File is {mb:.1f} MB — exceeds the {limit:.0f} MB limit.",
            code="FILE_TOO_LARGE",
        )


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------
def _extract_pdf(data: bytes) -> tuple[str, int]:
    """Return (text, page_count) from PDF bytes."""
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages), len(reader.pages)


def _extract_txt(data: bytes) -> tuple[str, int]:
    """Return (text, 1) from raw text bytes with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(encoding), 1
        except UnicodeDecodeError:
            continue
    raise IngestorError(
        "Unable to decode text file — unknown encoding.",
        code="ENCODING_ERROR",
    )


def _extract_docx(data: bytes) -> tuple[str, int]:
    """Return (text, paragraph_count) from DOCX bytes."""
    try:
        import docx  # python-docx — optional dep, imported lazily
    except ImportError:
        raise IngestorError(
            "python-docx is not installed. Run: pip install python-docx",
            code="MISSING_DEPENDENCY",
        )
    doc = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs), len(paragraphs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract(filename: str, data: bytes) -> IngestPayload:
    """
    Entry point for document extraction.

    Args:
        filename: Original filename (used to detect format).
        data:     Raw file bytes.

    Returns:
        IngestPayload with text, page_count, file_type, char_count.

    Raises:
        IngestorError: On any validation or extraction failure.
    """
    # 1. Extension check
    ext = _validate_extension(filename)

    # 2. Size check
    _validate_size(data)

    log.info("Extracting document", extra={"filename": filename, "bytes": len(data)})

    # 3. Format dispatch
    if ext == ".pdf":
        text, pages = _extract_pdf(data)
        file_type = "pdf"
    elif ext == ".txt":
        text, pages = _extract_txt(data)
        file_type = "txt"
    elif ext == ".docx":
        text, pages = _extract_docx(data)
        file_type = "docx"
    else:
        # Should never reach here after _validate_extension
        raise IngestorError("Unhandled extension.", code="INVALID_FILE_TYPE")

    # 4. Non-empty content check
    if not text or not text.strip():
        raise IngestorError(
            "No readable text found in document. "
            "Scanned PDFs without OCR are not supported.",
            code="EMPTY_CONTENT",
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
        text=text.strip(),
        page_count=pages,
        file_type=file_type,
        char_count=len(text),
    )
