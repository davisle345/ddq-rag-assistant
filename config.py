"""
Central configuration for the DDQ RAG Assistant.

All company-specific values, paths, and model settings live here and are
sourced from environment variables so the project can be re-pointed at any
organization's data without touching application code.

Copy `.env.example` to `.env` and adjust the values for your own deployment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Company / branding (used in prompts and answer phrasing)
# ---------------------------------------------------------------------------
# These default to a fictional company so the public repo ships with no
# real organization details. Override them in your `.env`.
COMPANY_NAME = os.getenv("COMPANY_NAME", "Acme Analytics, Inc.")
COMPANY_SHORT_NAME = os.getenv("COMPANY_SHORT_NAME", "Acme Analytics")
COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "acme.example")
COMPANY_FOUNDING_YEAR = int(os.getenv("COMPANY_FOUNDING_YEAR", "2010"))

# Aliases the model should treat as equivalent to the company name.
# Format: comma-separated "alias=canonical" pairs.
# Example: "acme=Acme Analytics,acme.example=Acme Analytics, Inc."
_raw_aliases = os.getenv("COMPANY_ALIASES", "")
COMPANY_ALIASES = {}
for _pair in _raw_aliases.split(","):
    if "=" in _pair:
        _alias, _canonical = _pair.split("=", 1)
        COMPANY_ALIASES[_alias.strip().lower()] = _canonical.strip()
if not COMPANY_ALIASES:
    COMPANY_ALIASES = {
        COMPANY_DOMAIN.lower(): COMPANY_SHORT_NAME,
        COMPANY_SHORT_NAME.lower(): COMPANY_SHORT_NAME,
    }

# ---------------------------------------------------------------------------
# Data / file paths
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_PATH = os.getenv(
    "KNOWLEDGE_BASE_PATH", "data/sample_ddq_knowledge_base.csv"
)
EXPORTS_DIR = os.getenv("EXPORTS_DIR", "ddq_exports")
POLICIES_DIR = os.getenv("POLICIES_DIR", "policies")

# ---------------------------------------------------------------------------
# Embedding + retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))

# ---------------------------------------------------------------------------
# LLM (AWS Bedrock)
# ---------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "us.meta.llama3-3-70b-instruct-v1:0"
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_NAME = os.getenv("SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))


def years_in_business() -> int:
    """Return how many years the company has been operating."""
    from datetime import datetime

    return datetime.now().year - COMPANY_FOUNDING_YEAR
