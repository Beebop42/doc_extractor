from utils import call_llm, parse_json, log
from schema import Attributes

INVOICE_PROMPT = """Extract structured data from this invoice. 
Respond ONLY in this JSON format:
{
    "seller_name": "entity selling service or goods. company or person name. Return string strictly",
    "seller_address": "seller's address, Return string strictly",
    "seller_phone": "seller's phone number. Only return the combination of +, -, 0-9 digital value",
    "seller_email": "seller's email. Return a string with @ in it strictly",
    "buyer_name": "also known as client etc. company or person name. Return string strictly",
    "buyer_address": "buyer's address. Return a string strictly",
    "buyer_phone": "buyer's phone number. Only return the combination of +, -, 0-9 digital value",
    "buyer_email": "buyer's email. Return a string with @ in it strictly",
    "invoice_date": "date of the invoice issuance",
    "items": [{"description": "", "quantity": "", "unit_price": ""}],
    "currency": "GBP/USD/JPY/EUR/CNY etc",
    "total_amount": "total amount. Return only the number, no other strings."
}"""

CHAT_PROMPT = """Extract structured data from this chat screenshot.
Respond ONLY in this JSON format:
{
    "chat_time":         "timestamp of the first message",
    "chat_date":         "date when the chat happened",
    "embedded_url":      ["any URLs found in messages"],
    "embedded_currency": ["the amount of money with curreny symbol found in message, e.g 500USD, £30 etc, return a list of string strictly"],
    "embedded_xfer":     "a confirmation message or a screenshot of money being transferred, return only Yes, No or Maybe",
    "otp_code":          "digital code consisting of 4-8 digits of numbers only. Used for account or payment verifications"
}"""

MARKETPLACE_LISTING_PROMPT = """Extract structured data from this marketplace screenshot.
Respond ONLY in this JSON format:
{
    "ecom_platform":    "which platform is the item listed, e.g. Amazon/Klarna/Facebook Market etc",
    "listed_time":      "when was the item listed, return strictly the following options: 1 week ago | 3 day ago | 1 days ago | other",
    "listed_item":      "an item that is being sold",
    "listed_item_desc": "a brief description of the item being sold",
    "listed_item_match":"Compare the item and the description in this screenshot to check if they match. Reply with exactly one word: Yes, No, or Maybe. Do not include any other text. e.g. description 'black honda' and a white car image should yield No",
    "listed_price":     "the price of the item. Return only the price number, no other strings.",
    "pic_contain_contact_info": "Check if the image of sold item containing any personal contact info, such as phone number, email address, social media account. Reply with exactly one word: Yes, No, or Maybe. Do not include any other text."
    "seller_location":  "where is the seller located",
    "seller_name":      "seller's name on this marketplace",
    "seller_acct_age":  "since which year the seller opened account on this market. Return only the year number, no other characters."
}
"""

WEBSITE_SCREENSHOT_PROMPT = """Extract structured data from this website screenshot.
Respond ONLY in this JSON format:
{
    "website_type":     "type of the website, e.g. gambling, traveling, shopping, bitcoin, banking etc",
    "website_err":      "does the screenshot contain error or warning message",
    "website_login":    "does the website contains login windows"      
}
"""


def parse(pages: list[str], 
          category: str, 
          model: str = "google/gemini-2.0-flash-001", 
          model_temperature: float=1.0,
          mime_type: str = "image/png") -> tuple[Attributes | None, dict]:
    """Extract structured attributes from pages based on predicted category.

    Uses a category-specific prompt and calls the LLM for each page.
    Results are merged across pages into a single `Attributes` object.

    Args:
        pages: List of base64-encoded images, one per page.
        category: Predicted category (e.g. `invoice`, `chat_screenshot`).
        model: OpenRouter model identifier.
        model_temperature: Sampling temperature.
        mime_type: MIME type for the embedded image payload.

    Returns:
        A tuple of `(attributes_or_none, stats)`. For unknown categories,
        returns `(None, {})`.
    """

    match category:
        case "invoice":
            prompt = INVOICE_PROMPT
        case "chat_screenshot":
            prompt = CHAT_PROMPT
        case "marketplace_listing":
            prompt = MARKETPLACE_LISTING_PROMPT
        case "website_screenshot":
            prompt = WEBSITE_SCREENSHOT_PROMPT
        case _:
            log.info("Category is unknown — route to human reviewer")
            return None, {}

    # Accumulate data across all pages.
    # `finish_reason` is not additive; keep the first non-empty value.
    all_stats = {"latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "finish_reason": ""}
    combined = {}

    for i, page in enumerate(pages):
        log.info(f"Parsing page {i + 1}/{len(pages)}...")
        # todo: this could go wrong
        raw, stats = call_llm(prompt, page, model=model, model_temperature=model_temperature, mime_type=mime_type) 
        # todo: this could go wrong
        data = parse_json(raw)

        for k in ["latency_ms", "prompt_tokens", "completion_tokens", "total_tokens"]:
            all_stats[k] += stats.get(k, 0)

        if not all_stats["finish_reason"]:
            fr = stats.get("finish_reason")
            if fr:
                all_stats["finish_reason"] = fr

        # Merge pages (extend lists, keep first non-null scalars)
        for key, val in data.items():
            if key not in combined:
                combined[key] = val
            elif isinstance(val, list):
                combined[key] = list(set(combined[key] + val))  # dedupe lists

    result = Attributes(
        event_type=category,
        seller_name=combined.get("seller_name"),
        seller_address=combined.get("seller_address"),
        seller_phone=combined.get("seller_phone"),
        seller_email=combined.get("seller_email"),
        buyer_name=combined.get("buyer_name"),
        buyer_address=combined.get("buyer_address"),
        buyer_phone=combined.get("buyer_phone"),
        buyer_email=combined.get("buyer_email"),
        invoice_date=combined.get("invoice_date"),
        items=combined.get("items"),
        currency=combined.get("currency"),
        total_amount=combined.get("total_amount"),
        chat_time=combined.get("chat_time"),
        chat_date=combined.get("chat_date"),
        embedded_url=combined.get("embedded_url"),
        embedded_currency=combined.get("embedded_currency"),
        embedded_xfer=combined.get("embedded_xfer"),
        otp_code=combined.get("otp_code"),
        ecom_platform=combined.get("ecom_platform"),
        listed_time=combined.get("listed_time"),
        listed_item=combined.get("listed_item"),
        listed_item_desc=combined.get("listed_item_desc"),
        pic_contain_contact_info=combined.get("pic_contain_contact_info"),
        listed_item_match=combined.get("listed_item_match"),
        listed_price=combined.get("listed_price"),
        seller_location=combined.get("seller_location"),
        seller_acct_age=combined.get("seller_acct_age"),
        website_type=combined.get("website_type"),
        website_err=combined.get("website_err"),
        website_login=combined.get("website_login"),
    )
    
    log.info(f"Parsing complete — category: {category}")
    return result, all_stats