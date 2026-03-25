from dataclasses import dataclass


@dataclass
class FileMetadata:
    filename:    str
    extension:   str
    size_kb:     float
    page_count:  int
    source_path: str


@dataclass
class ClassifierResult:
    category:   str        # "invoice" | "chat_screenshot" | "unknown"
    confidence: float      # 0.95 | 0.25 
    reason:     str


@dataclass
class Attributes:
    event_type:        str | None = None # "invoice" | "chat_screenshot" | "unknown"
    seller_name:       str | None = None
    seller_address:    str | None = None
    seller_phone:      str | None = None
    seller_email:      str | None = None
    buyer_name:        str | None = None
    buyer_address:     str | None = None
    buyer_phone:       str | None = None
    buyer_email:       str | None = None
    invoice_date:      str | None = None
    items:             list[dict] | None = None   # [{description, quantity, unit_price, total_price}]
    currency:          str | None = None
    total_amount:      str | None = None
    chat_time:         str | None = None
    chat_date:         str | None = None
    embedded_url:      list[str] | None = None
    embedded_currency: list[str] | None = None
    embedded_xfer:     str | None = None
    otp_code:          str | None = None
    ecom_platform:     str | None = None
    listed_time:       str | None = None         
    listed_item:       str | None = None
    listed_item_desc:  str | None = None
    listed_item_match: str | None = None
    pic_contain_contact_info: str | None = None
    listed_price:      str | None = None
    seller_location:   str | None = None
    seller_acct_age:   str | None = None
    website_type:      str | None = None
    website_err:       str | None = None
    website_login:     str | None = None

@dataclass
class ProcessingMetadata:
    model_used:          str
    latency_ms:          float | None=None
    prompt_tokens:       int | None=None
    completion_tokens:   int | None=None
    total_tokens:        int | None=None
    extraction_warnings: str | None=None


@dataclass
class PipelineResult:
    file_id:             str | None = None
    category:            str | None = None
    category_confidence: float | None = None
    extracted_fields:    Attributes | None = None
    scoring_rules:       list[str] | None = None
    risk_score:          float | None = None
    risk_label:          str | None = None
    summary:             str | None = None
    processing_metadata: ProcessingMetadata | None=None
