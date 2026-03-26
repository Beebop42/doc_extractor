from pathlib import Path
from schema import FileMetadata
from utils import log


SUPPORTED = {"jpg", "jpeg", "png", "pdf"}

def read_file(file_path: str) -> tuple[FileMetadata, bytes, str]:
    """Read and validate an input file, returning metadata and bytes.

    Args:
        file_path: Path to the input file.

    Returns:
        A tuple of `(metadata, raw_bytes, mime_type)`.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        ValueError: If the extension is unsupported, the file is empty, or
            the file magic bytes do not match the declared type.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: .{ext}. Supported types are {SUPPORTED}")

    mime_map = {
        "jpg": "image/jpeg", 
        "jpeg": "image/jpeg", 
        "png": "image/png",
        "pdf": "image/png", # pdf will be converted to png in preprocessing
    }

    with open(path, "rb") as f:
        raw_bytes = f.read()

    if len(raw_bytes) == 0:
        raise ValueError(f"File is empty: {path.name}")

    # Validate magic bytes
    magic = {
        "jpg":  b"\xff\xd8\xff",
        "jpeg": b"\xff\xd8\xff",
        "png":  b"\x89PNG",
        "pdf":  b"%PDF",
    }
    if not raw_bytes[:4].startswith(magic[ext]):
        raise ValueError(f"Invalid file signature: {path.name}")

    metadata = FileMetadata(
        filename=path.name,
        extension=ext,
        size_kb=round(len(raw_bytes) / 1024, 2),
        page_count=1,       # updated by preprocessor for PDFs
        source_path=str(path),
    )

    log.info(f"Read file: {path.name} ({metadata.size_kb} KB)")
    return metadata, raw_bytes, mime_map[ext]