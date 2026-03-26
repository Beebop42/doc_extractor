import base64
import fitz 
from schema import FileMetadata
from utils import log


def preprocess(metadata: FileMetadata, raw_bytes: bytes) -> tuple[list[str], FileMetadata]:
    """Convert input bytes into base64-encoded images for LLM usage.

    For images (`png`, `jpg`, `jpeg`), this returns a single base64 payload.
    For PDFs, it renders each page to a PNG and returns one base64 payload
    per page.

    Args:
        metadata: File metadata (including extension).
        raw_bytes: Raw file bytes read from disk.

    Returns:
        A tuple of `(pages, metadata)` where `pages` is a list of base64
        PNG-encoded strings and `metadata.page_count` is updated to the
        number of pages.
    """
    ext = metadata.extension

    if ext in ("jpg", "jpeg", "png"):
        pages = [base64.b64encode(raw_bytes).decode("utf-8")]

    elif ext == "pdf":
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            pages.append(encoded)
        doc.close()

    metadata.page_count = len(pages)
    log.info(f"Preprocessing complete: {len(pages)} page(s)")
    return pages, metadata