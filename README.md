# Doc Extractor

`doc_extractor` is a document and screenshot analysis pipeline for scam/fraud signal detection.
It takes an input file (`pdf`, `png`, `jpg`, `jpeg`), extracts structured attributes with an LLM through OpenRouter, applies rule-based scoring, and returns a risk result.

## What This Code Does

- Reads and validates uploaded files (images or PDFs).
- Converts each page into base64 image payloads for LLM processing.
- Classifies the file into one of:
  - `invoice`
  - `chat_screenshot`
  - `marketplace_listing`
  - `website_screenshot`
  - `unknown`
- Extracts category-specific fields into a structured schema.
- Runs fraud/scam scoring rules (e.g., OTP in chat, suspicious marketplace signals, archive-based recurrence checks).
- Persists extracted attributes and processing metadata into `archive/records.csv`.
- Returns a final pipeline result containing:
  - category and confidence
  - extracted fields
  - rules fired
  - risk score and risk label
  - token/latency metadata

## Pipeline

The core orchestration lives in `main.py` (`run_pipeline`):

1. **Read** (`reader.py`)
   - Validates extension and file signature.
   - Returns `FileMetadata`, raw bytes, and MIME type.
2. **Preprocess** (`preprocessor.py`)
   - Converts images directly to base64.
   - Renders each PDF page to PNG, then base64-encodes each page.
3. **Classify** (`classifier.py`)
   - Sends the first page to the LLM with a classification prompt.
   - Produces category + confidence.
4. **Parse** (`parser.py`)
   - Uses category-specific extraction prompts.
   - Merges multi-page outputs into a single `Attributes` object.
5. **Score** (`scorer.py`)
   - Executes rule set and computes risk probability/label.
6. **Archive** (`archive.py`)
   - Upserts one row in `archive/records.csv` for the processed file.
7. **Return Result** (`schema.py`)
   - Returns a structured `PipelineResult`.

## Prerequisites

- Python `>=3.13` (as defined in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenRouter API key

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Setup

Install dependencies and create/sync the virtual environment:

```bash
uv sync
```

## How to Run

### 1) CLI pipeline

Run the pipeline on a file path:

```bash
uv run python main.py /absolute/or/relative/path/to/file.pdf
```

Example:

```bash
uv run python main.py file_samples/invoice.pdf
```

Output is printed as JSON in the terminal, and archive records are written to:

- `archive/records.csv`

### 2) Streamlit app

Start the UI:

```bash
uv run streamlit run app.py
```

Then open the local URL shown in the terminal, upload a file, and click **Run Scam Check**.

## Project Structure (Key Files)

- `main.py` - end-to-end pipeline orchestrator
- `app.py` - Streamlit UI
- `reader.py` - file loading and validation
- `preprocessor.py` - image/PDF conversion to base64 pages
- `classifier.py` - category classification prompt call
- `parser.py` - category-specific structured extraction
- `scorer.py` - fraud rules and risk scoring
- `archive.py` - CSV persistence and historical lookups
- `schema.py` - dataclasses for metadata, attributes, and outputs
- `utils.py` - OpenRouter client, logging, JSON parsing helpers