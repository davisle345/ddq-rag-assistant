# Architecture

This document describes how the DDQ RAG Assistant is put together and the
reasoning behind the main design decisions.

## Overview

The system is a Retrieval-Augmented Generation (RAG) pipeline wrapped in a Gradio
web app. The core idea: rather than ask a language model to answer compliance and
security questions from its own (unreliable) memory, ground every answer in the
company's own previously vetted responses.

## Components

### 1. Configuration (`config.py`)
All environment-specific values - company branding, file paths, model id, AWS
region, server host/port - are read from environment variables with sensible
defaults. Nothing organization-specific is hard-coded in application logic, which
makes the project safe to open-source and easy to re-point at new data.

### 2. Knowledge base
A CSV of historical Q/A pairs with three meaningful columns: `Question`,
`Response`, and `Source`. The `Source` field follows a `Requester, Mon YYYY`
convention, which doubles as a recency signal.

### 3. Embeddings & vector store
On startup the app embeds every historical **question** with
`all-MiniLM-L6-v2` (a small, fast sentence-transformer) and builds an in-memory
**FAISS** index. Each vector carries the corresponding answer and source as
metadata. Retrieval returns the top-k (default 4) most semantically similar past
questions.

> Design note: the index is rebuilt on each launch. For larger knowledge bases,
> persist the FAISS index to disk and load it on startup.

### 4. Query preprocessing (`rag_enhancements.py`)
Before retrieval, each query is:
- spell-corrected for common questionnaire typos,
- normalized so company-name variants collapse to one canonical form,
- expanded so acronyms (SOC 2, HIPAA, GDPR, PII, MNPI) also match their
  long forms.

This meaningfully improves retrieval recall on real-world phrasing.

### 5. Template cache (`AnswerTemplateEngine`)
Frequently repeated questions (legal name, address, headcount, yes/no controls)
are matched against pre-built templates using fuzzy string similarity. A hit
returns the stored answer immediately, skipping the LLM call. This is the primary
cost/latency optimization and is surfaced in the Analytics tab.

### 6. Generation
On a cache miss, the retrieved Q/A pairs are formatted into a context block (each
annotated with a freshness indicator) and passed to the LLM through a LangChain
chain: `prompt | model | StrOutputParser`. The model runs on **AWS Bedrock**
(default Llama 3.3 70B Instruct) with a low temperature for deterministic,
factual output.

The prompt is deliberately strict. It instructs the model to:
- answer only from the provided context,
- speak in the first person ("we") as a compliance officer would,
- produce a **Supporting Evidence** section (cited sources) and a clean
  **Summary Answer**,
- never include "we don't have / not specified" hedging in the summary, since
  that text is meant for external submission.

### 7. Post-processing (`answer_cleaner.py` + `update_dynamic_dates`)
- `update_dynamic_dates` recalculates time-relative facts such as "years in
  business" so cached answers don't go stale.
- `AnswerCleaner` strips markdown, emoji, and source citations to produce the
  copy-ready summary shown in the UI.

### 8. Document scanner (`ddq_scanner.py`)
For batch work, the scanner extracts questions from an uploaded PDF or Word
document. It handles real-world document messiness: merged table cells, Word
content controls (form fields), section-header filtering, encoding fixes, and
de-duplication. Each extracted question is routed through the same
`ask_enhanced` pipeline, and results are exported to CSV.

> Limitation: question extraction is heuristic and depends on the source
> document's structure. Unusual layouts, scanned/image PDFs, or irregular tables
> can cause questions to be missed or merged. The scanner is best treated as a
> draft generator whose output is refined in the Chat tab, which is the primary,
> more reliable workflow.

## Request flow (Chat)

1. User submits a question.
2. `ask_enhanced` preprocesses it.
3. Template cache is checked; on a hit, return immediately.
4. Otherwise FAISS retrieves the top-k past Q/A pairs.
5. A confidence score is computed from source count, recency, and answer
   consistency.
6. Context is built and sent to the LLM.
7. Dynamic dates are corrected; a confidence banner is prepended.
8. The UI streams the full answer, extracts the clean summary, and attaches any
   relevant policy documents found in `policies/`.

## Trade-offs & notes

- **In-memory index**: simplest to run; rebuild cost grows with the knowledge
  base. Persist for scale.
- **Template fuzzy matching** uses `difflib.SequenceMatcher`, which is cheap and
  dependency-free but less precise than embedding similarity. The threshold is
  tuned conservatively (0.85) to avoid wrong cache hits.
- **Confidence scoring** is a transparent heuristic, not a calibrated
  probability; it is meant to prompt human review, not replace it.
