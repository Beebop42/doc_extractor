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