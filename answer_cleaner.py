"""
Answer Cleaner
Cleans and formats answers for Excel/CSV export
"""

import re


class AnswerCleaner:
    """Clean and format answers for export"""

    def clean_for_excel(self, answer: str) -> str:
        """
        Clean answer text for Excel export

        Removes:
        - Markdown formatting
        - Excessive whitespace
        - Special characters that break Excel

        Args:
            answer: Raw answer text

        Returns:
            Cleaned answer text
        """
        if not answer:
            return ""

        # Remove markdown headers (###, ##, #)
        answer = re.sub(r'^#{1,6}\s+', '', answer, flags=re.MULTILINE)

        # Remove markdown bold (**text** or __text__)
        answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)
        answer = re.sub(r'__(.+?)__', r'\1', answer)

        # Remove markdown italic (*text* or _text_)
        answer = re.sub(r'\*(.+?)\*', r'\1', answer)
        answer = re.sub(r'_(.+?)_', r'\1', answer)

        # Remove markdown links [text](url)
        answer = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', answer)

        # Remove markdown code blocks (```code```)
        answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL)

        # Remove inline code (`code`)
        answer = re.sub(r'`(.+?)`', r'\1', answer)

        # Remove horizontal rules (---, ***, ___)
        answer = re.sub(r'^[\-\*_]{3,}$', '', answer, flags=re.MULTILINE)

        # Remove emoji and special unicode characters
        answer = re.sub(r'[^\x00-\x7F]+', '', answer)

        # Clean up excessive whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = re.sub(r' {2,}', ' ', answer)

        # Remove leading/trailing whitespace
        answer = answer.strip()

        return answer

    def extract_summary(self, answer: str) -> str:
        """
        Extract just the summary answer portion, removing all metadata

        Args:
            answer: Full answer text with Supporting Evidence and Summary Answer

        Returns:
            Clean summary answer only, without confidence or source info
        """
        # Look for "Summary Answer:" section (case insensitive)
        match = re.search(r'\*\*Summary Answer\*\*:?\s*(.+?)(?=\n\n\*\*|$)', answer, re.DOTALL | re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
        else:
            # Try without bold formatting
            match = re.search(r'Summary Answer:?\s*(.+?)(?=\n\n|$)', answer, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
            else:
                # If no summary section found, return first paragraph
                paragraphs = answer.split('\n\n')
                summary = paragraphs[0].strip() if paragraphs else answer.strip()

        # Remove any remaining markdown formatting (multiple passes to catch nested/repeated)
        for _ in range(3):  # Multiple passes to catch all instances
            summary = re.sub(r'\*\*(.+?)\*\*', r'\1', summary, flags=re.DOTALL)  # Bold
            summary = re.sub(r'\*\*', '', summary)  # Remove any orphaned **
            summary = re.sub(r'__(.+?)__', r'\1', summary, flags=re.DOTALL)  # Bold alt
            summary = re.sub(r'\*(.+?)\*', r'\1', summary)  # Italic
            summary = re.sub(r'`(.+?)`', r'\1', summary)  # Code

        # Remove confidence indicators if present
        summary = re.sub(r'^[\u2705\u26a0\ufe0f\U0001f534]\s*.*?Confidence.*?\n+', '', summary, flags=re.MULTILINE)
        summary = re.sub(r'\(Confidence:.*?\)', '', summary, flags=re.IGNORECASE)

        # Remove source citations in brackets like [Company, Month Year]
        summary = re.sub(r'\[[\w\s,]+\d{4}\]', '', summary)

        # Clean up extra whitespace
        summary = re.sub(r'\n{2,}', '\n', summary)
        summary = re.sub(r' {2,}', ' ', summary)

        return summary.strip()

    def extract_confidence(self, answer: str) -> str:
        """
        Extract confidence level from answer

        Args:
            answer: Full answer text

        Returns:
            Confidence level (High, Medium, Low) or "N/A" if not found
        """
        # Look for confidence indicators in various formats
        patterns = [
            r'(?:High|Medium|Low)\s+Confidence',
            r'Confidence:\s*(High|Medium|Low)',
            r'\u2705.*?(High|Medium|Low)',
            r'\u26a0\ufe0f.*?(Medium)',
            r'\U0001f534.*?(Low)',
        ]

        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                # Extract the confidence level
                if 'High' in match.group(0):
                    return "High"
                elif 'Medium' in match.group(0):
                    return "Medium"
                elif 'Low' in match.group(0):
                    return "Low"

        # Check number of sources as fallback
        source_count = len(re.findall(r'\*\*Source\*\*:', answer))
        if source_count >= 3:
            return "High"
        elif source_count >= 2:
            return "Medium"
        elif source_count >= 1:
            return "Low"

        return "N/A"

    def remove_confidence_indicators(self, answer: str) -> str:
        """
        Remove confidence indicators from answer

        Args:
            answer: Answer text with confidence indicators

        Returns:
            Answer without confidence indicators
        """
        # Remove confidence lines (e.g., "High Confidence (85%)")
        answer = re.sub(r'^[\u2705\u26a0\ufe0f\U0001f534]\s*\*\*.*?Confidence.*?\*\*.*?\n+', '', answer, flags=re.MULTILINE)

        return answer.strip()
