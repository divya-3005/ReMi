import re
from collections import Counter
from typing import List, Set

def clean_text(raw_text: str) -> str:
    """Cleans raw text by removing excessive whitespace, fixing hyphenation,
    and stripping repeating headers/footers heuristically.

    Args:
        raw_text: Raw extracted text, where pages are separated by \\x0c.

    Returns:
        The cleaned text with page separators preserved.
    """
    # 1. Split into pages
    pages = raw_text.split("\x0c")
    if not pages or (len(pages) == 1 and not pages[0]):
        return ""

    # Helper to count words
    def word_count(line: str) -> int:
        return len(line.strip().split())

    # 2. Heuristic Header/Footer Detection
    # Collect candidate lines from the top 3 and bottom 3 non-empty lines of each page
    header_candidates: List[str] = []
    footer_candidates: List[str] = []

    page_lines_list: List[List[str]] = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines()]
        page_lines_list.append(lines)

        non_empty = [l for l in lines if l]

        # Top candidate lines (up to 3)
        top_lines = non_empty[:3]
        for tl in top_lines:
            if word_count(tl) < 5:
                header_candidates.append(tl)

        # Bottom candidate lines (up to 3)
        bottom_lines = non_empty[-3:]
        for bl in bottom_lines:
            if word_count(bl) < 5:
                footer_candidates.append(bl)

    # Find repeated headers/footers (occurrence >= 2 across pages)
    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)

    repeated_headers: Set[str] = {line for line, count in header_counts.items() if count >= 2}
    repeated_footers: Set[str] = {line for line, count in footer_counts.items() if count >= 2}

    # 3. Clean pages individually
    cleaned_pages: List[str] = []
    for lines in page_lines_list:
        new_lines: List[str] = []
        for line in lines:
            # Strip if matched as repeating header/footer
            if line in repeated_headers or line in repeated_footers:
                continue

            # Clean excessive spaces within the line
            cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
            new_lines.append(cleaned_line)

        page_text = "\n".join(new_lines)
        cleaned_pages.append(page_text)

    # Reassemble with page breaks
    cleaned_text = "\x0c".join(cleaned_pages)

    # 4. Fix broken hyphenation
    # Example: "re-\nsearch" -> "research"
    # Example: "re-\x0csearch" -> "re\x0csearch" (preserves page separator)
    def repl_hyphen(m: re.Match) -> str:
        w1, ws, w2 = m.group(1), m.group(2), m.group(3)
        if '\x0c' in ws:
            return f"{w1}\x0c{w2}"
        return f"{w1}{w2}"

    # Match a word, a hyphen, followed by space containing at least one newline/form-feed, then a word
    cleaned_text = re.sub(r'(\w+)-([ \t]*[\n\x0c][\s]*)(\w+)', repl_hyphen, cleaned_text)

    # 5. Clean overall excessive vertical whitespace
    # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraph structure)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    return cleaned_text.strip()
