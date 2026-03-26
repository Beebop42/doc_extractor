from utils import call_llm, parse_json, log
from schema import ClassifierResult

CLASSIFY_PROMPT = """Analyze this image and classify it as one of:
- "invoice"                 : a bill, receipt, or payment document
- "chat_screenshot"         : a messaging app screenshot (WhatsApp, iMessage, etc.)
- "marketplace_listing"     : an online marketplace or e-commerce listing screenshot (e.g. eBay, Amazon, Klarna, Facebook,etc.)
- "website_screenshot"      : a website screenshot (e.g. a blog post, a news article, a twitter mentioning, a gambling site, a webpage displaying 503 etc.)
- "unknown"                 : neither of the above

Respond ONLY in this JSON format:
{
  "category":   "invoice" | "chat_screenshot" | "marketplace_listing" | "website_screenshot" | "unknown",
  "confidence": float,
  "reason":     "brief explanation"
}"""


def classify(pages: list[str], 
             model: str = "google/gemini-2.0-flash-001", 
             model_temperature: float=1.0, 
             mime_type: str = "image/png") -> tuple[ClassifierResult, dict]:
    """Classify the input screenshot/document into a content category.

    The classifier sends the first page to the LLM using a JSON-only prompt
    and returns the predicted category (plus confidence) along with token
    usage statistics.

    Args:
        pages: List of base64 page images (one per page).
        model: OpenRouter model identifier.
        model_temperature: Sampling temperature.
        mime_type: MIME type used for the embedded image payload.

    Returns:
        A tuple of `(classifier_result, stats)`.
    """
    # Use the first page for classification
    raw, stats = call_llm(CLASSIFY_PROMPT, pages[0], model=model, model_temperature=model_temperature, mime_type=mime_type)
    data = parse_json(raw)

    result = ClassifierResult(
        category=data["category"],
        confidence=data["confidence"],
        reason=data["reason"],
    )

    log.info(f"Classified as : {result.category} (confidence: {result.confidence})")
    log.info(f"Reason        : {result.reason}")

    return result, stats