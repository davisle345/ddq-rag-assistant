"""
RAG Enhancement utilities for the DDQ Assistant.

These components improve retrieval accuracy and reduce LLM cost/latency around
the core RAG pipeline:

- QueryPreprocessor       normalize company names, fix typos, expand acronyms
- SourceFreshnessTracker  flag stale historical answers by parsing source dates
- AnswerTemplateEngine    serve high-confidence repeat answers without an LLM call
- SemanticDeduplicator    avoid storing duplicate Q&A when importing new data
- AnswerComparisonEngine  surface alternatives for medium-confidence matches
- BulkPDFImporter         parse Q&A pairs out of existing DDQ PDFs
"""

import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

import config


class QueryPreprocessor:
    """Normalize and enhance queries before retrieval."""

    def __init__(self):
        # Company name variations to normalize. Sourced from config so the
        # project ships with no real organization details baked into code.
        self.company_aliases = dict(config.COMPANY_ALIASES)

        # Common misspellings seen in real questionnaires
        self.spell_corrections = {
            'complience': 'compliance',
            'secutiry': 'security',
            'certfication': 'certification',
            'privicy': 'privacy',
            'personnally': 'personally',
            'identifyable': 'identifiable',
        }

    def normalize_company_names(self, query: str) -> str:
        """Normalize company name variations to a single canonical form."""
        query_lower = query.lower()
        for alias, canonical in self.company_aliases.items():
            query_lower = query_lower.replace(alias.lower(), canonical.lower())
        return query_lower

    def fix_common_typos(self, query: str) -> str:
        """Fix common spelling mistakes."""
        for typo, correct in self.spell_corrections.items():
            query = re.sub(r'\b' + typo + r'\b', correct, query, flags=re.IGNORECASE)
        return query

    def expand_abbreviations(self, query: str) -> str:
        """Expand common compliance/security acronyms to improve retrieval recall."""
        abbreviations = {
            r'\bSOC\s*2\b': 'SOC 2',
            r'\bGDPR\b': 'GDPR General Data Protection Regulation',
            r'\bHIPAA\b': 'HIPAA Health Insurance Portability and Accountability Act',
            r'\bPII\b': 'PII personally identifiable information',
            r'\bMNPI\b': 'MNPI material non-public information',
            r'\bAPI\b': 'API application programming interface',
        }

        for abbr, expansion in abbreviations.items():
            query = re.sub(abbr, expansion, query, flags=re.IGNORECASE)
        return query

    def preprocess(self, query: str) -> str:
        """Apply all preprocessing steps."""
        query = query.strip()
        query = self.fix_common_typos(query)
        query = self.normalize_company_names(query)
        query = self.expand_abbreviations(query)
        return query


class SourceFreshnessTracker:
    """Track and flag stale source data."""

    def __init__(self, stale_threshold_months: int = 6):
        self.stale_threshold_months = stale_threshold_months

    def parse_source_date(self, source: str) -> Optional[datetime]:
        """Extract a date from a source string like 'Northwind Capital, Apr 2025'."""
        month_patterns = [
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})',
            r'(\d{1,2})/(\d{4})',
        ]

        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        for pattern in month_patterns:
            match = re.search(pattern, source, re.IGNORECASE)
            if match:
                if '/' in pattern:
                    month, year = int(match.group(1)), int(match.group(2))
                else:
                    month_str = match.group(1).lower()[:3]
                    month = month_map.get(month_str, 1)
                    year = int(match.group(2))

                try:
                    return datetime(year, month, 1)
                except ValueError:
                    continue

        return None

    def is_stale(self, source: str) -> bool:
        """Check if source data is older than the stale threshold."""
        source_date = self.parse_source_date(source)
        if not source_date:
            return False  # Can't determine, assume fresh

        threshold_date = datetime.now() - timedelta(days=self.stale_threshold_months * 30)
        return source_date < threshold_date

    def get_freshness_indicator(self, source: str) -> str:
        """Return a human-readable freshness indicator for a source."""
        source_date = self.parse_source_date(source)
        if not source_date:
            return "Date unknown"

        months_old = (datetime.now() - source_date).days // 30

        if months_old < 3:
            return f"Recent ({source_date.strftime('%b %Y')})"
        elif months_old < 6:
            return f"{months_old} months old ({source_date.strftime('%b %Y')})"
        else:
            return f"Stale - {months_old} months old ({source_date.strftime('%b %Y')})"


class AnswerTemplateEngine:
    """Use templates for common question patterns to skip LLM calls."""

    def __init__(self, knowledge_base_path: str):
        self.kb_path = knowledge_base_path
        self.templates = self._build_templates()

    def _build_templates(self) -> Dict[str, Dict]:
        """Build answer templates from the knowledge base."""
        try:
            df = pd.read_csv(self.kb_path, encoding='utf-8').dropna()
        except UnicodeDecodeError:
            df = pd.read_csv(self.kb_path, encoding='latin-1').dropna()

        templates = {
            'yes_no': {
                'patterns': [
                    r'does\s+(?:the\s+)?(?:company|firm|organization)',
                    r'do\s+you\s+(?:have|provide|offer|maintain)',
                    r'is\s+(?:the\s+)?(?:company|data|system)',
                    r'are\s+(?:you|there)',
                    r'has\s+(?:the\s+)?company',
                ],
                'answers': {}
            },
            'contact_info': {
                'patterns': [
                    r'(?:contact|email|phone|address)',
                    r'who\s+(?:can|should)\s+(?:i|we)\s+contact',
                ],
                'answers': {}
            },
            'company_info': {
                'patterns': [
                    r'(?:legal\s+name|business\s+address|incorporation)',
                    r'how\s+many\s+(?:employees|people)',
                    r'what\s+(?:is|are)\s+(?:the\s+)?(?:company|business)',
                ],
                'answers': {}
            }
        }

        # Populate templates with actual answers from the knowledge base
        for _, row in df.iterrows():
            question = str(row['Question']).lower()
            answer = str(row['Response'])
            source = str(row['Source'])

            for template_type, template_data in templates.items():
                for pattern in template_data['patterns']:
                    if re.search(pattern, question, re.IGNORECASE):
                        key = self._normalize_question(question)
                        template_data['answers'][key] = {
                            'answer': answer,
                            'source': source,
                            'question': row['Question']
                        }
                        break

        return templates

    def _normalize_question(self, question: str) -> str:
        """Normalize a question for matching."""
        normalized = re.sub(r'[^\w\s]', '', question.lower())
        normalized = ' '.join(normalized.split())
        return normalized

    def find_template_match(self, query: str, threshold: float = 0.85) -> Optional[Dict]:
        """Find if a query closely matches a stored template answer."""
        query_normalized = self._normalize_question(query)

        best_match = None
        best_score = 0.0

        for template_type, template_data in self.templates.items():
            for stored_q, answer_data in template_data['answers'].items():
                score = SequenceMatcher(None, query_normalized, stored_q).ratio()

                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = {
                        'answer': answer_data['answer'],
                        'source': answer_data['source'],
                        'original_question': answer_data['question'],
                        'confidence': score,
                        'template_type': template_type,
                        'used_template': True
                    }

        return best_match


class SemanticDeduplicator:
    """Detect semantic duplicates before adding to the knowledge base."""

    def __init__(self, embeddings_model: HuggingFaceEmbeddings, threshold: float = 0.90):
        self.embeddings = embeddings_model
        self.threshold = threshold

    def find_duplicates(self, new_question: str, existing_questions: List[str]) -> List[Tuple[str, float]]:
        """Find semantically similar existing questions."""
        if not existing_questions:
            return []

        new_embedding = self.embeddings.embed_query(new_question)
        existing_embeddings = self.embeddings.embed_documents(existing_questions)

        duplicates = []
        for i, existing_emb in enumerate(existing_embeddings):
            similarity = self._cosine_similarity(new_embedding, existing_emb)
            if similarity >= self.threshold:
                duplicates.append((existing_questions[i], similarity))

        duplicates.sort(key=lambda x: x[1], reverse=True)
        return duplicates

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class AnswerComparisonEngine:
    """Show multiple possible answers for medium-confidence queries."""

    def __init__(self, confidence_threshold_low: float = 0.6, confidence_threshold_high: float = 0.8):
        self.low_threshold = confidence_threshold_low
        self.high_threshold = confidence_threshold_high

    def should_show_alternatives(self, confidence: float) -> bool:
        """Determine if alternative answers should be shown."""
        return self.low_threshold <= confidence < self.high_threshold

    def format_alternatives(self, docs: List, top_k: int = 3) -> str:
        """Format alternative answers for display."""
        if len(docs) < 2:
            return ""

        alternatives_text = "\n\n### Alternative Answers (Medium Confidence)\n"
        alternatives_text += "Please review these options and select the most appropriate:\n\n"

        for i, doc in enumerate(docs[:top_k], 1):
            alternatives_text += f"**Option {i}:**\n"
            alternatives_text += f"- **Answer:** {doc.metadata.get('answer', 'N/A')}\n"
            alternatives_text += f"- **Source:** {doc.metadata.get('source', 'N/A')}\n"
            alternatives_text += f"- **Original Q:** {doc.page_content}\n\n"

        return alternatives_text


class BulkPDFImporter:
    """Automated DDQ PDF ingestion and Q&A extraction."""

    def __init__(self, pdf_extractor_toolkit):
        self.pdf_toolkit = pdf_extractor_toolkit

    def extract_qa_pairs(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from a DDQ PDF."""
        extracted_text = self.pdf_toolkit.text_extractor()

        if not extracted_text:
            return []

        qa_pairs = []

        # Pattern 1: "Q001 ... <answer>" format
        pattern1 = r'Q\d+[:\s,]+(.+?)\n(.+?)(?=Q\d+|$)'
        matches1 = re.findall(pattern1, extracted_text, re.DOTALL)

        for question, answer in matches1:
            qa_pairs.append({
                'question': question.strip(),
                'answer': answer.strip(),
                'source': f'Imported from PDF - {datetime.now().strftime("%b %Y")}'
            })

        # Pattern 2: "Question: ... Answer: ..." format
        pattern2 = r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)'
        matches2 = re.findall(pattern2, extracted_text, re.DOTALL | re.IGNORECASE)

        for question, answer in matches2:
            qa_pairs.append({
                'question': question.strip(),
                'answer': answer.strip(),
                'source': f'Imported from PDF - {datetime.now().strftime("%b %Y")}'
            })

        return qa_pairs

    def append_to_knowledge_base(self, qa_pairs: List[Dict], kb_path: str) -> int:
        """Append extracted Q&A pairs to the knowledge base CSV."""
        if not qa_pairs:
            return 0

        df_existing = pd.read_csv(kb_path)
        df_new = pd.DataFrame(qa_pairs)
        df_new.columns = ['Question', 'Response', 'Source']

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(kb_path, index=False)

        return len(qa_pairs)
