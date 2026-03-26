import json
import logging
from datetime import datetime
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import pandas as pd
from schema import Attributes

log = logging.getLogger(__name__)

ARCHIVE_PATH = Path("archive/records.csv")


# ── Derive columns from dataclass ─────────────────────────────────────────
def get_attribute_columns() -> list[str]:
    """Return the list of attribute column names derived from `Attributes`.

    Returns:
        A list of dataclass field names for `schema.Attributes`.
    """
    return [f.name for f in fields(Attributes)]


# ── List fields — stored as JSON strings in CSV ────────────────────────────
LIST_FIELDS = {
    f.name for f in fields(Attributes)
    if "list" in str(f.type)
}  # → {"items", "embedded_url"}


# ── Pipeline metadata columns (not in Attributes) ─────────────────────────
META_COLUMNS = [
    "source_file",
    "page_count",
    "file_size_kb",
    "processed_at",
    "finish_reason",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

# Final column order: metadata first, then all Attributes fields
ALL_COLUMNS = META_COLUMNS + get_attribute_columns()


# ── Serialize Attributes to flat dict ─────────────────────────────────────
def attributes_to_dict(attrs: Attributes) -> dict:
    """Convert `Attributes` dataclass to a flat dictionary.

    List-typed attributes are serialized as JSON strings so they can be
    stored in a CSV file.

    Args:
        attrs: The `schema.Attributes` instance to serialize.

    Returns:
        A flat dictionary of attribute values suitable for writing to CSV.
    """
    result = {}
    for f in fields(attrs):
        val = getattr(attrs, f.name)
        if isinstance(val, list):
            result[f.name] = json.dumps(val) if val is not None else ""
        else:
            result[f.name] = val if val is not None else ""
    return result


# ── Deserialize row back to Attributes ────────────────────────────────────
def dict_to_attributes(row: dict) -> Attributes:
    """Reconstruct an `Attributes` dataclass instance from a CSV row.

    Args:
        row: A dictionary representing a CSV row (string values).

    Returns:
        An `Attributes` object constructed from the row contents.
    """
    kwargs = {}
    for f in fields(Attributes):
        val = row.get(f.name, "")
        if f.name in LIST_FIELDS:
            try:
                kwargs[f.name] = json.loads(val) if val else None
            except (json.JSONDecodeError, TypeError):
                kwargs[f.name] = None
        else:
            kwargs[f.name] = val if val != "" else None
    return Attributes(**kwargs)


# ── Load or create archive ─────────────────────────────────────────────────
def load_archive() -> pd.DataFrame:
    """Load the historical pipeline archive from `archive/records.csv`.

    If the file does not exist, returns an empty DataFrame with the expected
    columns.

    Returns:
        A pandas DataFrame containing historical records.
    """
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        df = pd.read_csv(ARCHIVE_PATH, dtype=str).fillna("")
        # Add any missing columns (e.g. schema evolved)
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[ALL_COLUMNS]
        log.info(f"Loaded archive: {len(df)} existing records")
    else:
        df = pd.DataFrame(columns=ALL_COLUMNS)
        log.info("No archive found — starting fresh")

    return df


def update_archive(
    source_file:  str,
    attrs:        Attributes,
    stats:        dict,
    page_count:   int   = 1,
    file_size_kb: float = 0.0,
) -> pd.DataFrame:
    """Insert or update a record in `archive/records.csv`.

    Args:
        source_file: Input file path (used as a stable record key).
        attrs: Extracted attributes for the file.
        stats: LLM processing statistics (latency/tokens/finish_reason).
        page_count: Number of pages derived from the input.
        file_size_kb: File size in kilobytes.

    Returns:
        The updated archive DataFrame.
    """

    df = load_archive()

    row = {
        "source_file":       source_file,
        "page_count":        page_count,
        "file_size_kb":      file_size_kb,
        "processed_at":      datetime.now().isoformat(),
        "finish_reason":     stats.get("finish_reason", "stop"),
        "latency_ms":        stats.get("latency_ms", 0),
        "prompt_tokens":     stats.get("prompt_tokens", 0),
        "completion_tokens": stats.get("completion_tokens", 0),
        "total_tokens":      stats.get("total_tokens", 0),
        **attributes_to_dict(attrs),
    }

    new_row = pd.DataFrame([row], columns=ALL_COLUMNS)

    if source_file in df["source_file"].values:
        log.info(f"Updating existing record: {source_file}")
        df = df[df["source_file"] != source_file]
    else:
        log.info(f"Appending new record: {source_file}")

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ARCHIVE_PATH, index=False)

    log.info(f"Archive saved → {ARCHIVE_PATH} ({len(df)} total records)")
    log.info(f"  Event type   : {attrs.event_type}")
    log.info(f"  Total tokens : {stats.get('total_tokens', 0)}")

    return df
