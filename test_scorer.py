import unittest
import pandas as pd
from archive import Attributes
from scorer import *
from unittest.mock import patch
from datetime import datetime, timedelta


class TestRuleOtpInChat(unittest.TestCase):

    # ── Should FIRE ───────────────────────────────────────────────────────
    
    def test_fires_with_otp_in_chat_screenshot(self):
        attrs = Attributes(event_type="chat_screenshot", otp_code="847291")
        result = rule_otp_in_chat(attrs)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 5)

    def test_fires_with_otp_containing_whitespace(self):
        """OTP with surrounding whitespace should still fire after strip."""
        attrs = Attributes(event_type="chat_screenshot", otp_code="  123456  ")
        result = rule_otp_in_chat(attrs)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 5)

    # ── Should NOT fire ───────────────────────────────────────────────────

    def test_does_not_fire_when_event_type_is_invoice(self):
        """OTP present but event is invoice — rule should not apply."""
        attrs = Attributes(event_type="invoice", otp_code="847291")
        result = rule_otp_in_chat(attrs)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


    def test_does_not_fire_when_otp_is_none(self):
        """OTP not present, rule is not fired"""
        attrs = Attributes(event_type="chat_screenshot", otp_code=None)
        result = rule_otp_in_chat(attrs)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


class TestRuleUrlTransactionInChat(unittest.TestCase):
    def test_fires_when_url_and_transaction_present_in_chat(self):
        """URL and transaction both present in chat screenshot, rule is fired"""
        attrs = Attributes(
            event_type    = "chat_screenshot",
            embedded_url  = ["https://example.com"],
            embedded_currency = "500 USD"
        )
        result = rule_url_transaction_in_chat(attrs)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 3)

    def test_does_not_fire_when_url_is_missing(self):
        """URL is missing from chat screenshot, rule is not fired"""
        attrs = Attributes(
            event_type    = "chat_screenshot",
            embedded_url  = ["N/A"],
            embedded_currency = "500 USD"
        )
        result = rule_url_transaction_in_chat(attrs)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


def make_archive(rows: list[dict]) -> pd.DataFrame:
    """Helper — build a minimal archive DataFrame for testing."""
    base = {
        "event_type":     "invoice",
        "seller_address": "123 Main St",
        "invoice_date":   datetime.now().isoformat(),
    }
    return pd.DataFrame([{**base, **r} for r in rows])

class TestRuleHighVolumeAddress(unittest.TestCase):

    # ── Should FIRE ───────────────────────────────────────────────────────

    @patch("scorer.load_archive")
    def test_fires_when_count_exceeds_threshold(self, mock_load):
        """
        Rule 002 triggered as more than 10 items (11) sold on invoice in p1d
        """
        mock_load.return_value = make_archive(
            [{"seller_address": "123 Main St"}] * 11
        )
        attrs = Attributes(event_type="invoice", seller_address="123 Main St")
        result = rule_high_volume_address(attrs)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 3)

    @patch("scorer.load_archive")
    def test_fires_with_custom_threshold(self, mock_load):
        """
        Rule 002 not triggered as fewer than 10 items (6) sold on invoice in p1d
        """
        mock_load.return_value = make_archive(
            [{"seller_address": "123 Main St"}] * 6
        )
        attrs = Attributes(event_type="invoice", seller_address="123 Main St")
        result = rule_high_volume_address(attrs, threshold=10)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)

    @patch("scorer.load_archive")
    def test_fires_with_custom_lookback(self, mock_load):
        """
        Rule 002 not triggered as fewer than 10 items (6) sold on invoice but not during p1d
        """
        past_3_days = (datetime.now() - timedelta(days=3)).isoformat()
        mock_load.return_value = make_archive(
            [{"seller_address": "123 Main St", "invoice_date": past_3_days}] * 11
        )
        attrs = Attributes(event_type="invoice", seller_address="123 Main St")
        result = rule_high_volume_address(attrs, threshold=10, lookback_days=1)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


    @patch("scorer.load_archive")
    def test_does_not_fire_when_archive_load_fails(self, mock_load):
        """
        Rule 002 not triggered due to error of reading archive
        """
        mock_load.side_effect = Exception("Disk error")
        attrs = Attributes(event_type="invoice", seller_address="123 Main St")
        result = rule_high_volume_address(attrs)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


class TestRuleMultiPhoneNrAddress(unittest.TestCase):

    @patch("scorer.load_archive")
    def test_address_has_more_than_threshold_phone_numbers(self, mock_load):
        """Address is linked to 6 unique phone numbers in past day, rule is fired"""
        mock_load.return_value = pd.DataFrame([
            {"event_type": "invoice", "seller_address": "123 Main St", "seller_phone": f"555-000{i}", "invoice_date": datetime.now().isoformat()}
            for i in range(6)
        ])
        attrs = Attributes(event_type="invoice", seller_address="123 Main St")
        result = rule_multi_phone_nr_address(attrs, threshold=5)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 4)

    @patch("scorer.load_archive")
    def test_address_has_fewer_phone_numbers_than_threshold(self, mock_load):
        """Address is linked to 2 unique phone numbers in past day, rule is not fired"""
        mock_load.return_value = pd.DataFrame([
            {"event_type": "invoice", "seller_address": "123 Main St", "seller_phone": "555-000", "invoice_date": datetime.now().isoformat()},
            {"event_type": "invoice", "seller_address": "123 Main St", "seller_phone": "N/A", "invoice_date": datetime.now().isoformat()},
            {"event_type": "invoice", "seller_address": "123 Main St", "seller_phone": None, "invoice_date": datetime.now().isoformat()},
            {"event_type": "invoice", "seller_address": "123 Main St", "seller_phone": "555-234", "invoice_date": datetime.now().isoformat()}
        ])
        attrs = Attributes(event_type="invoice", seller_address="123 Main St")
        result = rule_multi_phone_nr_address(attrs, threshold=5)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


class TestItemMismatch(unittest.TestCase):

    def test_item_pic_desc_mismatch(self):
        """Picture and Description in market listing do not match, rule fired"""
        attrs = Attributes(
            event_type    = "marketplace_listing",
            listed_item_match  = "No"
        )
        result = rule_item_mismatch_market(attrs)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 4)

    def test_item_pic_desc_maybe_match(self):
        """Picture and Description in market listing may match, rule not fired"""
        attrs = Attributes(
            event_type    = "marketplace_listing",
            listed_item_match  = "Maybe"
        )
        result = rule_item_mismatch_market(attrs)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)


class TestImgContainPII(unittest.TestCase):

    def test_img_contain_pii(self):
        attrs = Attributes(
            event_type = "marketplace_listing",
            pic_contain_contact_info = "Yes"
        )
        result = img_contain_pii_market(attrs)
        self.assertTrue(result.fired)
        self.assertEqual(result.score, 4)

    def test_img_not_contain_pii(self):
        attrs = Attributes(
            event_type = "marketplace_listing",
            pic_contain_contact_info = "N/A"
        )
        result = img_contain_pii_market(attrs)
        self.assertFalse(result.fired)
        self.assertEqual(result.score, 0)

if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()