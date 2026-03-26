import json
import tempfile
from pathlib import Path
import streamlit as st
from main import run_pipeline
from dataclasses import asdict

MODEL = 'google/gemini-2.0-flash-001'
TEMPERATURE = 0.2

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scam Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Scam Detection Pipeline")
st.caption("Upload an invoice or chat screenshot to analyse for fraud indicators.")

# ── File upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a file",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, PNG, JPG"
)

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file)
    else:
        st.info("PDF uploaded")

    if st.button("Run Scam Check", type="primary"):
        # Save upload to temp file so main.py can read it from disk
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.markdown(f"""
            **Model**: {MODEL}, **Model Temperature**: {TEMPERATURE}""")
        st.divider()

        with st.spinner("Analysing file..."):
            try:
                result = run_pipeline(tmp_path)
                result.file_id = tmp_path
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        # ── Risk badge ─────────────────────────────────────────────────────
        risk = result.risk_label.upper()
        colour = {
            "LOW":      "green",
            "MEDIUM":   "orange",
            "HIGH":     "red"
        }.get(risk, "gray")

        st.markdown(f"""
            <div style="padding:16px; border-radius:8px; background:{colour};
                        color:white; text-align:center; font-size:24px; font-weight:bold;">
                {risk} RISK — Score: {result.risk_score:.6f}
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Summary metrics ────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Category",            result.category)
        col2.metric("Category Confidence", result.category_confidence)
        col3.metric("Scam Score",          result.risk_score)

        st.divider()

        # ── Rules fired ────────────────────────────────────────────────────
        st.subheader("🚨 Rules Fired")
        if result.scoring_rules:
            st.error(f"**{result.scoring_rules}** — {result.summary}")
        else:
            st.success("No fraud rules triggered")

        st.divider()

        # ── Extracted fields ───────────────────────────────────────────────
        st.subheader("📋 Extracted Attributes")
        attrs = {k: v for k, v in asdict(result.extracted_fields).items() if v not in (None, "", [], {})}
        if attrs:
            for key, val in attrs.items():
                st.write(f"**{key}**: {val}")
        else:
            st.warning("No attributes extracted")

        st.divider()

        # ── LLM stats ──────────────────────────────────────────────────────
        with st.expander("LLM Stats"):
            stats = result.processing_metadata
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Latency (ms)",   stats.latency_ms)
            s2.metric("Prompt tokens",  stats.prompt_tokens)
            s3.metric("Output tokens",  stats.completion_tokens)
            s4.metric("Total tokens",   stats.total_tokens)

        # ── Raw JSON ───────────────────────────────────────────────────────
        with st.expander("Raw JSON result"):
            st.json(json.dumps(asdict(result), indent=2))