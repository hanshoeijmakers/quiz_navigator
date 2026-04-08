import re
from typing import List, Dict, Tuple

class PDFPreprocessor:
    """
    Preprocesses raw PDF text to improve LLM extraction accuracy.
    - Detects and marks questions, timeline sections, and task sections
    - Cleans noise and normalizes formatting
    - Returns enriched text with semantic markers
    """

    # Patterns for detection
    QUESTION_PATTERN = r'(?:^|\n)(?:Vraag|Q|Question)\s+(\d+)[\.:]*\s*(.+?)(?=\n(?:Vraag|Q|Question)\s+\d+|$)'
    CHAPTER_PATTERN = r'(?:^|\n)(?:Hoofdstuk|Chapter|H|Ch)\s+(\d+)[\.:]*\s*(.+?)(?=\n(?:Hoofdstuk|Chapter|H|Ch)\s+\d+|$)'

    # Time patterns: "20:30", "20.30", "14:00-16:00", "tussen 14 en 16 uur", etc.
    TIME_PATTERN = r'(?:\d{1,2}[:\.]?\d{2}(?:\s*-\s*\d{1,2}[:\.]?\d{2})?|tussen\s+\d{1,2}\s+(?:en|tot)\s+\d{1,2}\s+(?:uur|u))'

    # Task/action keywords
    TASK_KEYWORDS = [
        'maak', 'neem', 'foto', 'video', 'opname', 'film', 'knutsel', 'teken',
        'schets', 'schrijf', 'verzamel', 'verzamelen', 'bezoek', 'ga', 'gaan',
        'rij', 'rijden', 'fietsen', 'loop', 'lopen', 'sprint', 'zoek', 'zoeken',
        'onderzoek', 'vraag', 'vragen', 'interview', 'meet', 'weeg', 'tel', 'tellen',
        'noteer', 'record', 'stel', 'stellen', 'voer uit', 'uitvoeren', 'voltooi',
        'verzend', 'stuur', 'upload', 'presenteer', 'demonstreer', 'bouw', 'bouwen'
    ]

    def __init__(self):
        self.task_pattern = r'\b(?:' + '|'.join(self.TASK_KEYWORDS) + r')\b'

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw PDF text."""
        # Remove excessive newlines (more than 2 in a row)
        text = re.sub(r'\n\n\n+', '\n\n', text)

        # Remove common OCR artifacts and encoding issues
        text = text.replace('\\u00e9', 'é').replace('\\u00e8', 'è')

        # Replace page break markers - keep newline structure for question detection
        # This is important: "Vraag 1" might appear right after a page break
        text = re.sub(r'--- PAGINA \d+ ---', '\n', text)

        # Normalize whitespace (but preserve structure)
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single

        # Remove common header/footer patterns (repeated text patterns)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Remove very short repeated lines (likely headers/footers)
        # But be careful not to remove "Vraag X" which is short
        seen_short = {}
        filtered = []
        for line in lines:
            # Special case: don't filter out lines that look like question headers
            if re.match(r'^(?:Vraag|Q)\s+\d+', line, re.IGNORECASE):
                filtered.append(line)
            elif len(line) < 20:
                if line not in seen_short:
                    filtered.append(line)
                    seen_short[line] = 1
                elif seen_short[line] < 2:  # Allow up to 2 occurrences
                    filtered.append(line)
                    seen_short[line] += 1
            else:
                filtered.append(line)

        return '\n'.join(filtered)

    def detect_questions(self, text: str) -> List[Dict]:
        """
        Detect question blocks with enhanced OCR robustness.
        Tries multiple patterns to handle OCR errors and formatting variations.
        """
        questions = []

        # Multiple patterns to handle OCR errors and variations:
        patterns = [
            r'(?:^|\n)(Vraag|V[r1]?aag|Q)\s+(\d+)\s*[\.:—-]?\s*([^\n]*?)(?:\n|$)',  # Standard + OCR variants
            r'(?:^|\n)V\s*r\s*a\s*a\s*g\s+(\d+)',  # Spaced out letters (OCR error)
            r'(?:^|\n)(VRAAG|vraag)\s+(\d+)',  # Case variations
        ]

        found_questions = {}

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                # Extract question number - handle different group positions
                try:
                    if len(match.groups()) >= 2:
                        q_num = int(match.group(2))
                        q_text = match.group(3).strip() if len(match.groups()) >= 3 else ""
                    else:
                        q_num = int(match.group(1))
                        q_text = ""
                except (ValueError, IndexError):
                    continue

                # Skip if already found
                if q_num in found_questions:
                    continue

                # Extract surrounding context if minimal text
                if not q_text or len(q_text.strip()) < 10:
                    match_end = match.end()
                    following_text = text[match_end:match_end + 500]
                    following_lines = following_text.split('\n')[:3]
                    q_text = '\n'.join(following_lines).strip()

                    if not q_text or len(q_text.strip()) < 10:
                        q_text = "[Image-heavy page - text unclear]"

                # Infer chapter from context
                chapter = 1
                before_text = text[:match.start()]
                chapter_matches = re.findall(r'Hoofdstuk\s+(\d+)', before_text, re.IGNORECASE)
                if chapter_matches:
                    chapter = int(chapter_matches[-1])

                found_questions[q_num] = {
                    'num': q_num,
                    'chapter': chapter,
                    'text': q_text[:500],
                    'full_position': match.start()
                }

        questions = list(found_questions.values())
        return sorted(questions, key=lambda x: (x['chapter'], x['num']))

    def detect_timeline_sections(self, text: str) -> List[Dict]:
        """
        Detect sections that likely contain timeline information.
        Returns list of dicts with time pattern matches and surrounding context.
        """
        timeline_sections = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            # Check if line contains time pattern
            if re.search(self.TIME_PATTERN, line, re.IGNORECASE):
                # Extract context (2 lines before and after)
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = '\n'.join(lines[start:end])

                timeline_sections.append({
                    'line_num': i,
                    'text': line,
                    'context': context
                })

        return timeline_sections

    def detect_task_sections(self, text: str) -> List[Dict]:
        """
        Detect sections that likely contain task/doe-opdracht information.
        Returns list of dicts with action keyword matches and surrounding context.
        """
        task_sections = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            if re.search(self.task_pattern, line, re.IGNORECASE):
                # Extract context (1 line before and after)
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = '\n'.join(lines[start:end])

                task_sections.append({
                    'line_num': i,
                    'text': line,
                    'context': context
                })

        return task_sections

    def build_enriched_text(self, text: str, questions: List[Dict],
                           timeline_sections: List[Dict], task_sections: List[Dict]) -> str:
        """
        Build enriched text with semantic markers to guide LLM extraction.
        """
        lines = text.split('\n')
        enriched = []
        timeline_line_nums = {ts['line_num'] for ts in timeline_sections}
        task_line_nums = {ts['line_num'] for ts in task_sections}

        enriched.append("[FULL_TEXT_WITH_ANNOTATIONS]")

        for i, line in enumerate(lines):
            # Add timeline marker
            if i in timeline_line_nums:
                enriched.append("[CONTAINS_TIME_INFO]")

            # Add task marker
            if i in task_line_nums:
                enriched.append("[CONTAINS_TASK_INFO]")

            enriched.append(line)

        return '\n'.join(enriched)

    def create_extraction_summary(self, text: str, questions: List[Dict],
                                  timeline_sections: List[Dict], task_sections: List[Dict]) -> str:
        """
        Create a summary of detected structures to help LLM extraction.
        """
        summary = []
        summary.append("[EXTRACTION_SUMMARY]")
        summary.append(f"Detected Questions: {len(questions)}")

        if questions:
            summary.append("\nQuestion Structure Detected:")
            for q in questions[:10]:  # First 10
                summary.append(f"  - Vraag {q['num']} (Chapter {q['chapter']}): {q['text'][:100]}...")
            if len(questions) > 10:
                summary.append(f"  ... and {len(questions) - 10} more questions")

        summary.append(f"\nTimeline Sections Found: {len(timeline_sections)}")
        if timeline_sections:
            summary.append("Sample timeline markers:")
            for ts in timeline_sections[:5]:
                summary.append(f"  - Line {ts['line_num']}: {ts['text'][:80]}...")

        summary.append(f"\nTask Sections Found: {len(task_sections)}")
        if task_sections:
            summary.append("Sample task markers:")
            for ts in task_sections[:5]:
                summary.append(f"  - Line {ts['line_num']}: {ts['text'][:80]}...")

        summary.append("[/EXTRACTION_SUMMARY]\n")
        return '\n'.join(summary)

    def preprocess(self, raw_text: str) -> Dict:
        """
        Main preprocessing pipeline.
        Returns dict with cleaned text, detected structures, and enriched text for LLM.
        """
        # Step 1: Clean the text
        cleaned = self.clean_text(raw_text)

        # Step 2: Detect structures
        questions = self.detect_questions(cleaned)
        timeline_sections = self.detect_timeline_sections(cleaned)
        task_sections = self.detect_task_sections(cleaned)

        # Step 3: Build enriched text
        enriched = self.build_enriched_text(cleaned, questions, timeline_sections, task_sections)

        # Step 4: Create extraction summary
        summary = self.create_extraction_summary(cleaned, questions, timeline_sections, task_sections)

        return {
            'cleaned_text': cleaned,
            'questions_detected': questions,
            'timeline_sections_detected': len(timeline_sections),
            'task_sections_detected': len(task_sections),
            'enriched_text': enriched,
            'extraction_summary': summary,
            'full_text_for_llm': summary + enriched  # Combine summary + enriched for LLM
        }
