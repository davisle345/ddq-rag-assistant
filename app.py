"""
DDQ RAG Assistant - main application

A Retrieval-Augmented Generation assistant that turns a company's historical
Due Diligence Questionnaire (DDQ) and security questionnaire answers into an
interactive tool:

  - Chat tab:             ask a question, get a copy-ready answer with sources
  - Document Scanner tab: upload a blank DDQ, auto-answer every question
  - Analytics tab:        track template hit-rate and estimated cost savings

All company-specific settings live in `config.py` / `.env`.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from datetime import datetime

import config
from rag_enhancements import (
    QueryPreprocessor,
    SourceFreshnessTracker,
    AnswerTemplateEngine,
)
from ddq_scanner import DDQScanner

load_dotenv()

# ---------------------------------------------------------------------------
# Load the historical knowledge base
# ---------------------------------------------------------------------------
file_path = config.KNOWLEDGE_BASE_PATH

try:
    qa_data = pd.read_csv(file_path, encoding='utf-8').dropna()
except UnicodeDecodeError:
    qa_data = pd.read_csv(file_path, encoding='latin-1').dropna()

questions = qa_data["Question"].tolist()
answers = qa_data["Response"].tolist()
sources = qa_data["Source"].tolist()

# ---------------------------------------------------------------------------
# Embeddings + vector store
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
vector_store = FAISS.from_texts(
    texts=questions,
    embedding=embeddings,
    metadatas=[{"answer": answer, "source": source} for answer, source in zip(answers, sources)]
)
retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})

# Quick-win components
preprocessor = QueryPreprocessor()
freshness_tracker = SourceFreshnessTracker()
template_engine = AnswerTemplateEngine(file_path)

# ---------------------------------------------------------------------------
# LLM (AWS Bedrock)
# ---------------------------------------------------------------------------
# Credentials are read from the environment (.env). If they are not set,
# langchain/boto3 will fall back to the default AWS credential chain.
aws_model = ChatBedrockConverse(
    model=config.BEDROCK_MODEL_ID,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    temperature=config.LLM_TEMPERATURE,
    region_name=config.AWS_REGION,
)

# ---------------------------------------------------------------------------
# Prompt template (company values injected from config)
# ---------------------------------------------------------------------------
template = f"""
You are a specialized Q&A system designed to provide factual answers to user questions. Your primary goal is to extract and synthesize information from the provided 'Context' section, which consists of past Q/A pairs with sources formatted as [Company, Month Year].
Assume "{config.COMPANY_DOMAIN}" is equivalent to "{config.COMPANY_NAME}" or "{config.COMPANY_SHORT_NAME}" for questions and context.
Always answer using the word "we" as if you are the one providing the answer to the due diligence question on behalf of {config.COMPANY_SHORT_NAME}, as opposed to using terms like "the company" and "the most recent information does not specify..."
Provide answers like they're coming from a real professional compliance and security officer from our company.

Answer the user's question strictly based on the context below, which consists of past Q/A pairs with sources formatted as [Company, Month Year].

If the user's question is a yes/no question, start your Summary Answer explicitly with "Yes," or "No," as appropriate.
If the user's question asks for a fact, date, number, or other information that is not a yes/no question, provide a direct factual answer without starting with "Yes" or "No," and do not include any source attribution in the summary answer.

You may reference multiple Q/A pairs if:
- Each one contains information clearly relevant to the question.
- You combine them in a way that accurately reflects the content without fabricating or altering meaning.
- You clearly cite and quote each source used.

Instructions:
- Use only the context provided.
- You may refer to up to **three** Q/A pairs that are **most relevant and most recent**.
- Avoid soft or suggestive language such as "indicating," "suggesting," "appears," "it seems," or "we believe."
- Do **not** use phrases like "according to our sources" or "based on the data" - just state the answer directly.
- Do **not** include source citations or company names inside the summary answer.
- If the question asks for an exact number or percentage that is not provided directly in the data, do **not** infer or estimate the answer.
- If multiple relevant answers exist, prioritize the one with the **most recent date** (e.g., Apr 2024 over Mar 2023), but you may include older answers if they provide useful context or complementary details.
- Each Q/A pair must be reproduced verbatim and clearly cited.

**CRITICAL RULE FOR SUMMARY ANSWER - READ CAREFULLY:**
The Summary Answer is for external submission to third parties. It must ONLY contain positive information that answers the question.

**FORBIDDEN PHRASES - NEVER USE THESE IN SUMMARY ANSWER:**
- "However, specific details on..."
- "We do not specify..."
- "The most recent information does not specify..."
- "Details are not provided..."
- "Information is not available..."
- "We do not mention..."
- "It is not clear..."
- "The context does not include..."
- Any variation of "we don't have" or "not specified" or "not provided"

**RULE FOR MULTI-PART QUESTIONS:**
If a question asks multiple things (e.g., "Do you do X? If so, who does it and what topics?"), answer ONLY the parts you have information for. Simply omit the parts you cannot answer. Do NOT say "we don't specify who" or similar.

**CORRECT EXAMPLE:**
Question: "Does the Company conduct annual compliance training? If so, who conducts it and what topics were covered?"
Context shows: "Yes, we conduct annual training covering HIPAA, Privacy, and Data Security."
CORRECT Summary Answer: "Yes, we conduct annual compliance training covering topics including HIPAA, Privacy and Data Security Awareness, FCRA, MNPI, and Data Privacy Training."
WRONG Summary Answer: "Yes, we conduct annual compliance training covering HIPAA, Privacy, and Data Security. However, we do not specify who conducts the training."

**ANOTHER CORRECT EXAMPLE:**
Question: "What is your data retention policy and who approves exceptions?"
Context shows: "We retain data for 7 years in accordance with regulatory requirements."
CORRECT Summary Answer: "We retain data for 7 years in accordance with regulatory requirements."
WRONG Summary Answer: "We retain data for 7 years in accordance with regulatory requirements. However, information about who approves exceptions is not specified."

Structure your response as follows:
1. **Supporting Evidence**: For each Q/A pair used (max 3), include:
- **Source**: [Company, Month Year]
    - **Q**: ...
    - **A**: ...
    - **Explanation**: Briefly explain why this Q/A pair is relevant.
2. **Summary Answer**: Provide a direct, factual answer synthesized from up to 3 sources in a neutral, professional tone. Use "we" when speaking instead of third-person pronouns like "the company." Include "Yes," or "No," at the beginning of the sentence only if the question is yes/no. Do not include any source references or company names in the summary answer. ONLY include positive information that directly answers what you CAN answer. Never mention missing information or unanswered parts of the question.

If no information is found or the answer cannot be derived AT ALL, respond with: "Not given in the dataset. [Brief explanation on why the data is missing or not available]."

Do not infer or generate information that is not present in the context.

Context:
{{context}}

User's Question: {{question}}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | aws_model | StrOutputParser()

# Statistics tracking
stats = {
    'total_queries': 0,
    'template_hits': 0,
    'llm_calls': 0,
    'cost_saved': 0.0
}


def update_dynamic_dates(response, question):
    """
    Post-process a response to update dynamic date calculations such as
    "years in business" so answers stay correct over time.
    """
    import re

    FOUNDING_YEAR = config.COMPANY_FOUNDING_YEAR
    current_year = datetime.now().year
    years_in_business = current_year - FOUNDING_YEAR

    question_lower = question.lower()
    is_years_question = any(phrase in question_lower for phrase in [
        'how many years',
        'years in business',
        'years has the company been',
        'how long has the company',
        'company age',
        'years of operation',
        'operating for how many years'
    ])

    if not is_years_question:
        return response

    year_patterns = [
        r'\b(\d{1,2})\s+years?\b',
        r'\bfor\s+(\d{1,2})\s+years?\b',
        r'\bover\s+(\d{1,2})\s+years?\b',
        r'\bmore than\s+(\d{1,2})\s+years?\b',
    ]

    updated_response = response
    for pattern in year_patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        for match in matches:
            old_years = int(match.group(1))
            # Only update values close to the expected figure to avoid false positives
            if abs(old_years - years_in_business) <= 5 and old_years < years_in_business:
                old_text = match.group(0)
                new_text = old_text.replace(str(old_years), str(years_in_business))
                updated_response = updated_response.replace(old_text, new_text, 1)

    return updated_response


def ask_enhanced(question):
    """Enhanced ask function with query preprocessing, template caching, and confidence scoring."""
    stats['total_queries'] += 1

    # Step 1: Preprocess query
    processed_query = preprocessor.preprocess(question)

    # Step 2: Check template match first (cost optimization)
    template_match = template_engine.find_template_match(processed_query)

    if template_match:
        stats['template_hits'] += 1
        stats['cost_saved'] += 0.02  # Estimated cost per LLM call avoided

        freshness = freshness_tracker.get_freshness_indicator(template_match['source'])
        updated_answer = update_dynamic_dates(template_match['answer'], processed_query)

        response = f"""### Quick Answer (Template Match)

**Summary Answer:** {updated_answer}

**Source:** {template_match['source']}
**Freshness:** {freshness}
**Confidence:** {template_match['confidence']:.1%}

---
*This answer was retrieved from templates, saving processing time and cost.*
"""
        return response

    # Step 3: Retrieve from vector store
    stats['llm_calls'] += 1
    docs = retriever.invoke(processed_query)

    # Step 4: Calculate confidence based on retrieval quality
    num_sources = len(docs)
    recent_sources = sum(1 for doc in docs if any(year in doc.metadata['source'] for year in ['2024', '2025', '2026']))
    recency_score = recent_sources / max(num_sources, 1)

    unique_answers = len(set(doc.metadata['answer'][:50].lower() for doc in docs[:3]))
    consistency_score = 1.0 if unique_answers == 1 else 0.7 if unique_answers == 2 else 0.5  # noqa: F841

    base_confidence = 0.6
    source_bonus = min(num_sources * 0.1, 0.3)
    recency_bonus = recency_score * 0.1

    confidence = min(base_confidence + source_bonus + recency_bonus, 0.95)
    confidence = max(confidence, 0.5)

    # Step 5: Format context with freshness indicators
    context_parts = []
    for doc in docs:
        source = doc.metadata['source']
        freshness = freshness_tracker.get_freshness_indicator(source)
        context_parts.append(
            f"Source: {source} {freshness}\n"
            f"Exact Q/A: '{doc.page_content}' -> '{doc.metadata['answer']}'"
        )
    context = "\n\n".join(context_parts)

    # Step 6: Generate response
    response = chain.invoke({"context": context, "question": processed_query})

    # Step 6.5: Post-process for dynamic date calculations
    response = update_dynamic_dates(response, processed_query)

    # Step 7: Add a confidence indicator
    if confidence >= 0.85:
        conf_emoji = "[High]"
        conf_label = "High Confidence"
        conf_detail = f"Based on {num_sources} consistent source(s)"
    elif confidence >= 0.70:
        conf_emoji = "[Good]"
        conf_label = "Good Confidence"
        conf_detail = f"Based on {num_sources} source(s) - recommend review"
    else:
        conf_emoji = "[Low]"
        conf_label = "Lower Confidence"
        conf_detail = f"Limited sources ({num_sources}) - verify independently"

    response = f"{conf_emoji} **{conf_label}** - {conf_detail}\n\n{response}"

    return response


def get_stats():
    """Return usage statistics for the Analytics tab."""
    if stats['total_queries'] == 0:
        return "No queries processed yet."

    template_rate = (stats['template_hits'] / stats['total_queries']) * 100

    return f"""### Usage Statistics

- **Total Queries:** {stats['total_queries']}
- **Template Hits:** {stats['template_hits']} ({template_rate:.1f}%)
- **LLM Calls:** {stats['llm_calls']}
- **Estimated Cost Saved:** ${stats['cost_saved']:.2f}
- **Cache Hit Rate:** {template_rate:.1f}%
"""


def detect_relevant_documents(user_question, summary_answer):
    """
    Detect which supporting policy documents are relevant to a question/answer
    and return any that exist on disk in the configured policies directory.

    Drop your own policy PDFs into the `policies/` folder using the filenames
    below (see policies/README.md) to have them auto-attached in the Chat tab.
    """
    documents = []
    doc_names = []
    user_lower = user_question.lower()
    summary_lower = summary_answer.lower()

    # Generic policy mapping. Each entry can map to one OR more files, so a
    # single topic (e.g. SOC 2) can attach multiple documents. Drop your own
    # PDFs into the policies/ folder using these filenames (see
    # policies/README.md) to have them auto-attached in the Chat tab.
    doc_keywords = {
        "privacy policy": {
            "files": ["privacy-policy.pdf"],
            "name": "Privacy Policy",
            "keywords": ["privacy policy", "privacy"]
        },
        "terms of service": {
            "files": ["terms-of-service.pdf"],
            "name": "Terms of Service",
            "keywords": ["terms of service", "terms", "tos"]
        },
        "consent to use": {
            "files": ["consent-to-use-of-data.pdf"],
            "name": "Consent to Use of Data",
            "keywords": ["consent to use", "consent", "use of data"]
        },
        "insider trading": {
            "files": ["insider-trading-policy.pdf"],
            "name": "Insider Trading Policy",
            "keywords": ["insider trading", "mnpi", "material non-public", "material, non-public", "personal trading", "trading policy", "code of ethics"]
        },
        "information security": {
            "files": ["information-security-policy.pdf"],
            "name": "Information Security Policy (InfoSec)",
            "keywords": ["information security", "infosec", "security policy"]
        },
        "incident response": {
            "files": ["incident-response-policy.pdf"],
            "name": "Incident Response Policy",
            "keywords": ["incident response", "incident management", "breach response", "security incident", "breach notification"]
        },
        "soc 2": {
            "files": ["soc2-executive-summary.pdf", "soc2-technical-report.pdf"],
            "name": "SOC 2 Report",
            "keywords": ["soc 2", "soc2", "soc 1", "soc 3", "service organization control"]
        },
        "data privacy third party": {
            "files": ["data-privacy-third-party-use-policy.pdf"],
            "name": "Data Privacy and Third-Party Use Policy",
            "keywords": ["data privacy", "third-party", "third party", "data gathering", "rights and permissions"]
        },
        "de-identification": {
            "files": ["de-identification-guide.pdf"],
            "name": "De-Identification Guide",
            "keywords": ["de-identification", "deidentification", "nistir", "confidentiality"]
        },
        "hipaa": {
            "files": ["hipaa-compliance.pdf"],
            "name": "HIPAA Compliance",
            "keywords": ["hipaa", "health information", "phi", "protected health"]
        },
        "baa": {
            "files": ["business-associate-agreement.pdf"],
            "name": "Business Associate Agreement (BAA)",
            "keywords": ["business associate agreement", "baa", "business associate"]
        },
        "cybersecurity": {
            "files": ["cybersecurity-policy.pdf"],
            "name": "Cybersecurity Policy",
            "keywords": ["cybersecurity", "cyber security", "security measures", "security breach", "security risks"]
        },
        "data security": {
            "files": ["data-security-overview.pdf"],
            "name": "Data Security Overview",
            "keywords": ["data security", "encryption", "data destruction", "data sharing", "data ownership", "cloud security"]
        },
    }

    # A topic matches if any of its keywords appear in the question or answer.
    # When matched, every existing file for that topic is attached.
    for doc_key, doc_info in doc_keywords.items():
        for keyword in doc_info["keywords"]:
            if keyword in user_lower or keyword in summary_lower:
                attached_any = False
                for file_name in doc_info["files"]:
                    filepath = os.path.join(config.POLICIES_DIR, file_name)
                    if os.path.exists(filepath) and filepath not in documents:
                        documents.append(filepath)
                        attached_any = True
                if attached_any and doc_info["name"] not in doc_names:
                    doc_names.append(doc_info["name"])
                break  # matched this topic; move on to the next

    return documents, doc_names


def chat_interface(user_input, chat_history):
    """Gradio chat handler with a streaming, ChatGPT/Gemini-style response."""
    import time
    import re

    if not user_input.strip():
        yield "", chat_history, "", []
        return

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"]
    casual_phrases = ["how are you", "what's up", "whats up", "sup", "thanks", "thank you", "bye", "goodbye"]

    user_lower = user_input.lower().strip()
    is_greeting = any(user_lower == greeting or user_lower.startswith(greeting + " ") for greeting in greetings)
    is_casual = any(phrase in user_lower for phrase in casual_phrases)

    if is_greeting or (is_casual and len(user_input.split()) < 10):
        friendly_response = """Hello! I'm your DDQ Assistant, here to help with Due Diligence Questionnaire inquiries.

I can answer questions about:
- Security and compliance (SOC 2, HIPAA, etc.)
- Privacy policies and data protection
- Terms of service and legal documents
- Information security practices
- And much more from your historical DDQ data!

What would you like to know?"""

        chat_history = chat_history + [(user_input, friendly_response)]
        yield "", chat_history, "", []
        return

    # Show an animated "thinking" indicator
    thinking_html = '<div class="thinking-indicator"><div class="spinner"></div><span>Thinking...</span></div>'
    chat_history = chat_history + [(user_input, thinking_html)]
    yield "", chat_history, "", []

    time.sleep(0.3)

    response = ask_enhanced(user_input)

    # Clean up "Not given in the dataset" responses
    if "not given in the dataset" in response.lower():
        clean_response = re.sub(r'\*\*Supporting Evidence\*\*:.*?(?=\*\*Summary Answer\*\*:|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
        clean_response = re.sub(r'\[(?:High|Good|Low)\].*?Confidence.*?\n', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\*\*Summary Answer\*\*:?\s*', '', clean_response, flags=re.IGNORECASE)
        clean_response = clean_response.strip()
        response = clean_response if clean_response else response

    # Reset bubble, then stream
    chat_history[-1] = (user_input, "")
    yield "", chat_history, "", []

    tokens = re.findall(r'\S+|\s+', response)
    streamed_response = ""
    word_count = 0

    for token in tokens:
        streamed_response += token
        if token.strip():
            word_count += 1
            if word_count % 2 == 0:
                chat_history[-1] = (user_input, streamed_response)
                yield "", chat_history, "", []
                time.sleep(0.008)
        else:
            chat_history[-1] = (user_input, streamed_response)
            yield "", chat_history, "", []

    # Extract the summary answer for the copy-ready box
    summary_match = re.search(r'\*\*Summary Answer\*\*:?\s*(.+?)(?=\n\n\*\*|$)', response, re.DOTALL | re.IGNORECASE)
    if not summary_match:
        summary_match = re.search(r'Summary Answer:?\s*(.+?)(?=\n\n|$)', response, re.DOTALL | re.IGNORECASE)

    summary_text = summary_match.group(1).strip() if summary_match else response

    clean_summary = summary_text
    clean_summary = re.sub(r'^\[(?:High|Good|Low)\]\s*.*?Confidence.*?\n+', '', clean_summary, flags=re.MULTILINE)
    clean_summary = re.sub(r'\(Confidence:.*?\)', '', clean_summary, flags=re.IGNORECASE)
    clean_summary = re.sub(r'Confidence:.*?\n', '', clean_summary, flags=re.IGNORECASE)
    clean_summary = re.sub(r'\[[\w\s,]+\d{4}\]', '', clean_summary)

    for _ in range(3):
        clean_summary = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_summary, flags=re.DOTALL)
        clean_summary = re.sub(r'\*\*', '', clean_summary)
        clean_summary = re.sub(r'__(.+?)__', r'\1', clean_summary, flags=re.DOTALL)
        clean_summary = re.sub(r'\*(.+?)\*', r'\1', clean_summary)
        clean_summary = re.sub(r'_(.+?)_', r'\1', clean_summary)
        clean_summary = re.sub(r'`(.+?)`', r'\1', clean_summary)

    clean_summary = re.sub(r'^#{1,6}\s+', '', clean_summary, flags=re.MULTILINE)
    clean_summary = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean_summary)
    clean_summary = re.sub(r'\n{3,}', '\n\n', clean_summary)
    clean_summary = re.sub(r' {2,}', ' ', clean_summary)
    clean_summary = clean_summary.strip()

    relevant_docs, doc_names = detect_relevant_documents(user_input, clean_summary)

    response_with_docs = response
    if doc_names:
        doc_list = ", ".join(doc_names)
        response_with_docs += f"\n\n**Attached Documents:** {doc_list}"

    chat_history[-1] = (user_input, response_with_docs)
    yield "", chat_history, clean_summary, relevant_docs


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
custom_css = """
/* Main container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}

/* Header styling */
.header-container {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}

/* Info boxes - Light mode */
.info-box-light {
    padding: 1rem;
    background: #f8fafc;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.info-box-light p {
    margin: 0;
    color: #64748b;
    font-size: 0.95rem;
}

/* Info boxes - Warning style */
.info-box-warning {
    margin-top: 1rem;
    padding: 0.75rem;
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 4px;
}

.info-box-warning strong {
    color: #92400e;
}

/* Info boxes - Success style */
.info-box-success {
    margin-top: 1.5rem;
    padding: 1rem;
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 4px;
}

.info-box-success strong {
    color: #166534;
}

/* Dark mode overrides */
.dark .info-box-light {
    background: rgba(51, 65, 85, 0.5) !important;
}

.dark .info-box-light p {
    color: #cbd5e1 !important;
}

.dark .info-box-warning {
    background: rgba(254, 243, 199, 0.15) !important;
    border-left-color: #f59e0b !important;
}

.dark .info-box-warning strong {
    color: #fbbf24 !important;
}

.dark .info-box-success {
    background: rgba(240, 253, 244, 0.15) !important;
    border-left-color: #22c55e !important;
}

.dark .info-box-success strong {
    color: #4ade80 !important;
}

/* Chat container - seamless */
.chatbot-container {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.chatbot {
    border: none !important;
    background: transparent !important;
}

[class*="chatbot"] {
    border: none !important;
}

/* User message styling - with border */
.user {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 18px !important;
    padding: 8px 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    line-height: 1.5 !important;
    overflow: visible !important;
}

.user p {
    margin: 0 !important;
    padding: 0 !important;
}

.dark .user {
    background: rgba(55, 65, 81, 0.5) !important;
    border-color: #4b5563 !important;
}

/* Bot message styling - no border */
.bot {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 10px 0 !important;
    overflow: visible !important;
}

.bot p {
    margin: 0 !important;
    padding: 0 !important;
}

.user, .bot, .message {
    overflow: visible !important;
    max-height: none !important;
}

.prose {
    overflow: visible !important;
    max-height: none !important;
}

/* Thinking/Loading animation */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.thinking-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #667eea;
    font-size: 14px;
    padding: 8px 0;
}

.dark .thinking-indicator {
    color: #93c5fd;
}

.spinner {
    width: 20px;
    height: 20px;
    border: 3px solid rgba(102, 126, 234, 0.2);
    border-top-color: #667eea;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

.thinking-indicator span {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Gemini-style input row */
.chat-input-row {
    gap: 8px !important;
    align-items: flex-end !important;
    margin-top: 1rem !important;
}

.gemini-input textarea {
    border-radius: 24px !important;
    border: 1px solid #e5e7eb !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    resize: none !important;
    min-height: 52px !important;
    max-height: 156px !important;
    overflow-y: auto !important;
}

.dark .gemini-input textarea {
    border-color: #4b5563 !important;
    background: rgba(31, 41, 55, 0.5) !important;
}

.gemini-input textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: transform 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;
    padding: 0 !important;
    font-size: 20px !important;
    color: white !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    flex-shrink: 0 !important;
}

.send-btn:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
}

/* Tabs */
.tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Section headers */
.section-header-blue {
    padding: 1.5rem;
    background: linear-gradient(135deg, #e0e7ff 0%, #cffafe 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

.section-header-blue h3 {
    margin: 0 0 0.5rem 0;
    color: #1e40af;
    font-size: 1.5rem;
}

.section-header-yellow {
    padding: 1.5rem;
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

.section-header-yellow h3 {
    margin: 0 0 0.5rem 0;
    color: #92400e;
    font-size: 1.5rem;
}

/* Info card */
.info-card {
    margin-top: 1rem;
    padding: 1rem;
    background: #f1f5f9;
    border-radius: 8px;
}

.dark .info-card {
    background: rgba(51, 65, 85, 0.4) !important;
}

/* Scrollable textbox */
.scroll-text textarea {
    overflow-y: auto !important;
    max-height: 600px !important;
}
"""

with gr.Blocks(title="DDQ Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div class="header-container">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">DDQ Assistant</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                AI-powered Due Diligence Questionnaire automation
            </p>
        </div>
    """)

    with gr.Tab("Chat"):
        gr.HTML("""
        <div class="info-box-light">
            <p>
                Ask questions about security, compliance, certifications, or any DDQ topic.
                Powered by your historical DDQ data.
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=550,
                    show_label=False,
                    container=False,
                    elem_classes=["chatbot-container"],
                    bubble_full_width=False
                )

                with gr.Row(elem_classes=["chat-input-row"]):
                    user_input = gr.Textbox(
                        placeholder="Ask me anything about your DDQ data...",
                        show_label=False,
                        scale=20,
                        container=False,
                        lines=1,
                        max_lines=8,
                        autofocus=True,
                        elem_classes=["gemini-input"]
                    )
                    submit = gr.Button(">", scale=1, variant="primary", elem_classes=["send-btn"], min_width=50)

                gr.HTML("""
                <div class="info-box-warning">
                    <strong>Smart Features:</strong> Auto spell-check - Company name normalization - Source freshness tracking - Template answers
                </div>
                """)

            with gr.Column(scale=1):
                gr.HTML("""
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin-bottom: 1rem;">
                    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #667eea;">Summary Answer</h3>
                    <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Clean answer ready to copy</p>
                </div>
                """)

                summary_output = gr.Textbox(
                    label="",
                    placeholder="Summary will appear here after each response...",
                    lines=8,
                    max_lines=15,
                    interactive=True,
                    show_label=False
                )

                copy_btn = gr.Button("Copy to Clipboard", variant="primary", size="sm")

                gr.HTML("""
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; margin: 1.5rem 0 1rem 0;">
                    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #667eea;">Supporting Documents</h3>
                    <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Download relevant policies</p>
                </div>
                """)

                doc_files = gr.File(
                    label="",
                    file_count="multiple",
                    interactive=False,
                    show_label=False
                )

        def copy_to_clipboard(text):
            return "Copied!"

        def reset_copy_button():
            import time
            time.sleep(2)
            return "Copy to Clipboard"

        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[summary_output],
            outputs=[copy_btn],
            js="(text) => {navigator.clipboard.writeText(text); return text;}"
        ).then(
            fn=reset_copy_button,
            outputs=[copy_btn]
        )

        submit.click(chat_interface, inputs=[user_input, chatbot], outputs=[user_input, chatbot, summary_output, doc_files])
        user_input.submit(chat_interface, inputs=[user_input, chatbot], outputs=[user_input, chatbot, summary_output, doc_files])

    with gr.Tab("Document Scanner"):
        scanner = DDQScanner()

        gr.HTML("""
        <div class="section-header-blue">
            <h3>Automated Document Processing</h3>
            <p>
                Upload your DDQ document and let AI automatically answer all questions.
                Supports PDF and Word formats.
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload DDQ Document (PDF or Word)",
                    file_types=[".pdf", ".docx", ".doc"],
                    file_count="single"
                )

                process_btn = gr.Button(
                    "Process Document",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"]
                )

            with gr.Column(scale=1):
                gr.HTML(f"""
                <div class="info-card">
                    <strong>What happens:</strong><br/>
                    - Questions are automatically extracted<br/>
                    - AI answers each question<br/>
                    - Results saved to <code>{config.EXPORTS_DIR}/</code><br/>
                    - Includes confidence levels
                </div>
                """)

        scanner_summary_output = gr.Markdown(label="Results Summary")

        with gr.Accordion("View Full Results", open=False):
            text_output = gr.Textbox(
                label="Detailed Answers",
                lines=25,
                max_lines=None,
                show_label=False,
                interactive=False
            )

        csv_output = gr.File(label="Download CSV File")

        def process_uploaded_file(file):
            if file is None:
                return "Please upload a file.", None, None

            try:
                output_dir = config.EXPORTS_DIR
                os.makedirs(output_dir, exist_ok=True)

                with open(file.name, 'rb') as f:
                    file_bytes = f.read()

                results = scanner.process_document_from_bytes(
                    file_bytes,
                    file.name,
                    ask_enhanced
                )

                if not results:
                    return "No questions found in the document.", None, None

                text_output_str = scanner.format_results_as_text(results)

                csv_filename = f"ddq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                scanner.export_to_csv(results, csv_path)

                success_count = sum(1 for r in results if r['status'] == 'success')
                summary = f"""
### Processing Complete

- **Total Questions Found:** {len(results)}
- **Successfully Answered:** {success_count}
- **Errors:** {len(results) - success_count}
- **Saved to:** `{csv_path}`

Download the CSV file below for the complete results.
"""
                return summary, text_output_str, csv_path

            except Exception as e:
                return f"Error processing file: {str(e)}", None, None

        process_btn.click(
            fn=process_uploaded_file,
            inputs=[file_input],
            outputs=[scanner_summary_output, text_output, csv_output]
        )

    with gr.Tab("Analytics"):
        gr.HTML("""
        <div class="section-header-yellow">
            <h3>Performance Analytics</h3>
            <p>
                Track usage, cost savings, and system performance metrics.
            </p>
        </div>
        """)

        stats_display = gr.Markdown(get_stats())

        with gr.Row():
            refresh_btn = gr.Button("Refresh Stats", variant="secondary", size="lg")

        refresh_btn.click(lambda: get_stats(), outputs=stats_display)

        gr.HTML("""
        <div class="info-box-success">
            <strong>Cost Optimization:</strong> Template answers bypass expensive LLM calls, saving time and money while maintaining accuracy.
        </div>
        """)

if __name__ == "__main__":
    demo.launch(server_name=config.SERVER_NAME, server_port=config.SERVER_PORT)
