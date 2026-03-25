import json
import sys
from dataclasses import asdict
from reader import read_file
from preprocessor import preprocess
from classifier import classify
from parser import parse
from utils import log
from archive import update_archive
from scorer import score
from schema import PipelineResult, ProcessingMetadata


def run_pipeline(file_path: str, model: str = "google/gemini-2.0-flash-001", model_temperature: float=1.0) -> PipelineResult:
    log.info("═" * 50)
    log.info(f"Starting pipeline for: {file_path}")

    # 1. Read
    metadata, raw_bytes, mime_type = read_file(file_path)

    # 2. Preprocess
    pages, metadata = preprocess(metadata, raw_bytes)

    # 3. Classify
    classifier_result, classify_stats = classify(pages, model=model, model_temperature=model_temperature, mime_type=mime_type)

    # 4. Parse
    parsed_data, parse_stats = parse(pages, classifier_result.category, model=model, mime_type=mime_type)

    # 5. run scorring 
    score_result = score(parsed_data, temp=1.0)

    # 6. Combine stats
    combined_stats = {
        k: classify_stats.get(k, 0) + parse_stats.get(k, 0)
        for k in ["latency_ms", "prompt_tokens", "completion_tokens", "total_tokens"]
    }
    combined_stats["finish_reason"] = parse_stats.get("finish_reason", "stop")
    update_archive(
        source_file  = file_path,
        attrs        = parsed_data,
        stats        = combined_stats,
        page_count   = metadata.page_count,
        file_size_kb = metadata.size_kb,
    )

    result = PipelineResult(
        category=classifier_result.category,
        category_confidence=classifier_result.confidence,
        extracted_fields=parsed_data,
        scoring_rules=score_result.rules_fired,
        risk_score=score_result.risk_score,
        risk_label=score_result.risk_level,
        summary=score_result.summary,
        processing_metadata=ProcessingMetadata(
            model_used=model,
            latency_ms=combined_stats["latency_ms"],
            prompt_tokens=combined_stats["prompt_tokens"],
            completion_tokens=combined_stats["completion_tokens"],
            total_tokens=combined_stats["total_tokens"],
            extraction_warnings = None if combined_stats["finish_reason"]=="stop" else combined_stats["finish_reason"] 
        )
    )

    total_stats = {
        "classify": classify_stats,
        "parse":    parse_stats,
        "total_tokens": (
            classify_stats.get("total_tokens", 0) +
            parse_stats.get("total_tokens", 0)
        )
    }
    log.info(f"Pipeline complete — total tokens: {total_stats['total_tokens']}")
    log.info("═" * 50)
    return result


if __name__ == "__main__":
    file_path = sys.argv[1] 
    model = "google/gemini-2.0-flash-001"
    model_temperature = 0.2
    result = run_pipeline(file_path, model, model_temperature)
    result.file_id = file_path

    print(json.dumps(asdict(result), indent=2))