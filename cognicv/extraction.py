"""Text extraction from uploaded files (PDF, DOCX, TXT/MD)."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


class ExtractionError(Exception):
    """Raised when a file cannot be converted to usable text."""


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ExtractionError("PDF is password-protected.")
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc

    # Fallback for stubborn layouts if pdfplumber is installed (optional dep)
    if len(text.strip()) < 100:
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                alt = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if len(alt.strip()) > len(text.strip()):
                text = alt
        except ImportError:
            pass
        except Exception:
            pass
    return text


def _extract_docx(data: bytes) -> str:
    try:
        import docx2txt

        return docx2txt.process(io.BytesIO(data)) or ""
    except ImportError:
        raise ExtractionError(
            "DOCX support requires the `docx2txt` package (pip install docx2txt)."
        )
    except Exception as exc:
        raise ExtractionError(f"Could not read DOCX: {exc}") from exc


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text(filename: str, data: bytes) -> str:
    """Convert file bytes to normalized plain text.

    Raises ExtractionError for unsupported types or unreadable files.
    """
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _extract_pdf(data)
    elif name.endswith(".docx"):
        text = _extract_docx(data)
    elif name.endswith((".txt", ".md")):
        text = _extract_txt(data)
    else:
        raise ExtractionError(
            f"Unsupported file type: {filename!r}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
