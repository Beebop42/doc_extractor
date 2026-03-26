import logging
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from archive import Attributes, load_archive
import numpy as np

log = logging.getLogger(__name__)


# ── Score result ───────────────────────────────────────────────────────────
@dataclass
class ScoreResult:
    risk_score:    float
    risk_level:    str          # "low" | "medium" | "high" 
    rules_fired:   list[str]    # which rules triggered
    summary:       str          # brief explanation why this score is given
    breakdown:     list[dict]   # per-rule detail


# ── Individual rule result ─────────────────────────────────────────────────
@dataclass
class RuleResult:
    rule_id:     str
    description: str
    fired:       bool
    score:       int


# ── Risk level thresholds ──────────────────────────────────────────────────
def get_risk_level(score: float) -> str:
    """Convert a numeric risk score into a discrete risk label.

    Args:
        score: Total numeric score computed from fired rules.

    Returns:
        One of: `low`, `medium`, or `high`.
    """
    if score == 0:
        return "low"
    elif score <= 3:
        return "medium"
    else:
        return "high"


# ── Rules ──────────────────────────────────────────────────────────────────

def rule_otp_in_chat(attrs: Attributes) -> RuleResult:
    """Rule 1: Detect OTP codes in chat screenshots.

    Args:
        attrs: Extracted attributes for the content.

    Returns:
        A `RuleResult` with `fired=True` and score 5 when an OTP is found.
    """
    RULE_ID = "RULE_001"
    fired = (
        attrs.event_type == "chat_screenshot" and
        attrs.otp_code is not None and
        attrs.otp_code.strip() != "" and 
        attrs.otp_code.strip() != "N/A"
    )
    return RuleResult(
        rule_id     = RULE_ID,
        description = "OTP code shared in chat screenshot",
        fired       = fired,
        score       = 5 if fired else 0
    )


def rule_url_transaction_in_chat(attrs: Attributes) -> RuleResult:
    """Rule 2: Detect URL + currency/transaction signals in chat screenshots.

    Args:
        attrs: Extracted attributes for the content.

    Returns:
        A `RuleResult` fired when `embedded_url` and `embedded_currency`
        contain valid (non-empty, non-`N/A`) values.
    """
    def check_if_valid(var: list|str) -> bool:
        """Check whether a value is present and not a sentinel.

        Args:
            var: Either a list of strings or a single string value.

        Returns:
            True if at least one non-empty, non-`N/A` value is present.
        """
        valid = False 
        if isinstance(var, list):
            valid_urls = len([x for x in var if (x is not None) and (x.strip() not in ['N/A', ''])])
            if valid_urls > 0:
                valid = True
        else:
            if var is not None and var.strip() not in ['', 'N/A']:
                valid = True
        return valid 

    RULE_ID = "RULE_002"
    has_valid_url = check_if_valid(attrs.embedded_url)
    has_valid_currency = check_if_valid(attrs.embedded_currency)
    fired = (
        attrs.event_type == "chat_screenshot" and
        has_valid_url and
        has_valid_currency
    )
    return RuleResult(
        rule_id     = RULE_ID,
        description = "URL and transaction are mentioned in chat screenshot",
        fired       = fired,
        score       = 3 if fired else 0
    )

# ── Rule 3 ─────────────────────────────────────────────────────────────────
def rule_high_volume_address(
    attrs:         Attributes,
    threshold:     int = 10,
    lookback_days: int = 1,
) -> RuleResult:
    """Rule 3: Seller address appears on many invoices within a lookback window.

    Args:
        attrs: Extracted attributes for the content.
        threshold: Minimum count (strictly greater than this threshold fires).
        lookback_days: Lookback window in days.

    Returns:
        A `RuleResult` fired when the seller address matches more than
        `threshold` invoice records in the archive within `lookback_days`.
    """
    RULE_ID     = "RULE_003"
    DESCRIPTION = f"Seller address linked to >{threshold} invoices in past {lookback_days}d"

    def no_fire() -> RuleResult:
        """Return a non-fired result for this rule.

        Returns:
            A `RuleResult` with `fired=False` and score `0`.
        """
        return RuleResult(
            rule_id     = RULE_ID,
            description = DESCRIPTION,
            fired       = False,
            score       = 0
        )

    # ── Gate 1: correct event type ─────────────────────────────────────────
    if attrs.event_type != "invoice":
        return no_fire()

    # ── Gate 2: address present ────────────────────────────────────────────
    if not attrs.seller_address or not attrs.seller_address.strip():
        return no_fire()

    # ── Gate 3: archive must load successfully ─────────────────────────────
    try:
        df = load_archive()
    except Exception as e:
        log.warning(f"{RULE_ID}: Archive load failed — {e}")
        return no_fire()

    if df.empty:
        return no_fire()

    # ── Gate 4: count matching address within lookback window ──────────────
    try:
        cutoff = datetime.now() - timedelta(days=lookback_days)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df = df.dropna(subset=["invoice_date"])
        mask = (
            (df["event_type"]     == "invoice") &
            (df["invoice_date"]   >= cutoff) &
            (df["seller_address"].str.strip().str.lower()
             == attrs.seller_address.strip().lower())
        )

        count = len(df[mask])
    except Exception as e:
        log.warning(f"{RULE_ID}: Archive query failed — {e}")
        return no_fire()

    # ── Only fire when count strictly exceeds threshold ────────────────────
    if count > threshold:
        log.info(f"{RULE_ID}: FIRED — '{attrs.seller_address}' has {count} invoices in past {lookback_days}d")
        return RuleResult(
            rule_id     = RULE_ID,
            description = DESCRIPTION,
            fired       = True,
            score       = 3,
        )

    log.info(f"{RULE_ID}: skipped — '{attrs.seller_address}' has {count} invoices (threshold: {threshold})")
    return no_fire()


# ── Rule 4 ─────────────────────────────────────────────────────────────────
def rule_multi_phone_nr_address(
    attrs:         Attributes,
    threshold:     int = 5,
    lookback_days: int = 1,
) -> RuleResult:
    """Rule 4: Seller address appears with many distinct phone numbers.

    Args:
        attrs: Extracted attributes for the content.
        threshold: Minimum unique phone number count (strictly greater fires).
        lookback_days: Lookback window in days.

    Returns:
        A `RuleResult` fired when the seller address is associated with
        more than `threshold` unique seller phone numbers within the window.
    """
    RULE_ID     = "RULE_004"
    DESCRIPTION = f"Seller address linked to >{threshold} invoices phone numbers in past {lookback_days}d"

    def no_fire() -> RuleResult:
        """Return a non-fired result for this rule.

        Returns:
            A `RuleResult` with `fired=False` and score `0`.
        """
        return RuleResult(
            rule_id     = RULE_ID,
            description = DESCRIPTION,
            fired       = False,
            score       = 0
        )

    # ── Gate 1: correct event type ─────────────────────────────────────────
    if attrs.event_type != "invoice":
        return no_fire()

    # ── Gate 2: address present ────────────────────────────────────────────
    if not attrs.seller_address or not attrs.seller_address.strip():
        return no_fire()

    # ── Gate 3: archive must load successfully ─────────────────────────────
    try:
        df = load_archive()
    except Exception as e:
        log.warning(f"{RULE_ID}: Archive load failed — {e}")
        return no_fire()

    if df.empty:
        return no_fire()

    # ── Gate 4: count matching address within lookback window ──────────────
    try:
        cutoff = datetime.now() - timedelta(days=lookback_days)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df["seller_phone"] = df["seller_phone"].str.strip()
        df["seller_phone"] = df["seller_phone"].replace('', None)
        df["seller_phone"] = df["seller_phone"].replace('N/A', None)
        df = df.dropna(subset=["invoice_date", "seller_phone"])
        mask = (
            (df["event_type"]     == "invoice") &
            (df["invoice_date"]   >= cutoff) &
            (df["seller_address"].str.strip().str.lower()
             == attrs.seller_address.strip().lower()) 
        )

        count = df[mask]["seller_phone"].nunique()
    except Exception as e:
        log.warning(f"{RULE_ID}: Archive query failed — {e}")
        return no_fire()

    # ── Only fire when count strictly exceeds threshold ────────────────────
    if count > threshold:
        log.info(f"  {RULE_ID}: FIRED — '{attrs.seller_address}' has {count} invoices in past {lookback_days}d")
        return RuleResult(
            rule_id     = RULE_ID,
            description = DESCRIPTION,
            fired       = True,
            score       = 4,
        )

    log.info(f"  {RULE_ID}: skipped — '{attrs.seller_address}' has {count} invoices (threshold: {threshold})")
    return no_fire()


# ── Rule 5 ─────────────────────────────────────────────────────────────────
def rule_item_mismatch_market(attrs: Attributes) -> RuleResult:
    """Rule 5: Marketplace listing picture/description mismatch.

    Args:
        attrs: Extracted attributes for the content.

    Returns:
        A `RuleResult` fired when `listed_item_match` is explicitly `No`.
    """
    RULE_ID = "RULE_005"
    fired = (
        attrs.event_type == "marketplace_listing" and
        attrs.listed_item_match is not None and 
        attrs.listed_item_match.strip() == 'No'
    )
    return RuleResult(
        rule_id     = RULE_ID,
        description = "Item picture and description on market place don't match",
        fired       = fired,
        score       = 4 if fired else 0
    )


# ── Rule 6 ─────────────────────────────────────────────────────────────────
def img_contain_pii_market(attrs: Attributes) -> RuleResult:
    """Rule 6: Marketplace image appears to contain PII.

    Args:
        attrs: Extracted attributes for the content.

    Returns:
        A `RuleResult` fired when `pic_contain_contact_info` is `Yes`.
    """
    RULE_ID = "RULE_006"
    fired = (
        attrs.event_type == "marketplace_listing" and
        attrs.pic_contain_contact_info.strip() == 'Yes'
    )
    return RuleResult(
        rule_id     = RULE_ID,
        description = "Item picture contains PII info",
        fired       = fired,
        score       = 4 if fired else 0
    )


# ── Rule registry ──────────────────────────────────────────────────────────
RULES = [
    rule_otp_in_chat,
    rule_url_transaction_in_chat,
    rule_high_volume_address,
    rule_multi_phone_nr_address,
    rule_item_mismatch_market,
    img_contain_pii_market
]


# ── Scorer ─────────────────────────────────────────────────────────────────
def score(attrs: Attributes, temp: float=1.0) -> ScoreResult:
    """Compute a final risk score and risk label from extracted attributes.

    Args:
        attrs: Extracted attributes for the input content.
        temp: Temperature parameter used to convert total rule score into a
            risk probability-like value.

    Returns:
        A `ScoreResult` containing `risk_score`, `risk_level`, `rules_fired`,
        and a human-readable summary.
    """
    log.info("── Fraud Scoring ───────────────────────")

    results      = [rule(attrs) for rule in RULES]
    rules_fired  = [r.rule_id for r in results if r.fired]
    summary      = "\n".join([r.description for r in results if r.fired])
    total_score  = sum(r.score for r in results)
    proba        = 1.0 - np.exp(-total_score/temp)
    risk_level   = get_risk_level(total_score)

    breakdown = [
        {
            "rule_id":     r.rule_id,
            "description": r.description,
            "fired":       r.fired,
            "score":       r.score
        }
        for r in results
    ]

    for r in results:
        status = "FIRED" if r.fired else "skipped"
        log.info(f"  [{status}] {r.rule_id}: (score: +{r.score})")

    log.info(f"  Total score : {total_score}")
    log.info(f"  Risk level  : {risk_level.upper()}")


    return ScoreResult(
        risk_score    = proba,
        risk_level    = risk_level,
        rules_fired   = rules_fired,
        summary       = summary,
        breakdown     = breakdown,
    )