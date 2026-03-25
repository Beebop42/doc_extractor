import base64
import fitz 
from schema import FileMetadata
from utils import log


def preprocess(metadata: FileMetadata, raw_bytes: bytes) -> tuple[list[str], FileMetadata]:
    """Return list of base64 PNG images (one per page)."""
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