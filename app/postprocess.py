from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
import re


@dataclass
class CitationChunk:
    text: str
    source_file: str
    page_number: int


BULLET_RE = re.compile(r"^\s*-\s+")
CITATION_TOKEN_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
LOW_VALUE_PATTERNS = [
    re.compile(r"\bsource\b", re.IGNORECASE),
    re.compile(r"\|\s*page\s*:", re.IGNORECASE),
    re.compile(r"\bnone used\b", re.IGNORECASE),
    re.compile(r"^\s*\[\d+\]\s*[^.]+p\.\d+\s*$", re.IGNORECASE),
]


def extract_citation_ids(text: str) -> Set[int]:
    ids: Set[int] = set()
    for raw in CITATION_TOKEN_RE.findall(text):
        for part in raw.split(","):
            value = part.strip()
            if value.isdigit():
                ids.add(int(value))
    return ids


def extract_bullets(answer: str) -> List[str]:
    lines = answer.splitlines()
    bullets: List[str] = []
    in_answer = False
    saw_answer_heading = False

    for ln in lines:
        stripped = ln.strip()
        if stripped.lower() == "answer:":
            in_answer = True
            saw_answer_heading = True
            continue
        if stripped.lower() == "cited sources:":
            if in_answer:
                break
            continue
        if in_answer and BULLET_RE.match(ln):
            bullets.append(stripped)

    # Fallback: if model skipped headings, use all bullet lines.
    if not saw_answer_heading:
        bullets = [ln.strip() for ln in lines if BULLET_RE.match(ln)]

    return bullets


def is_low_value_bullet(bullet: str) -> bool:
    # Ignore pure citation/source listing bullets that do not answer the question.
    content = CITATION_TOKEN_RE.sub("", bullet)
    content = re.sub(r"^\s*-\s*", "", content).strip()
    if not content:
        return True
    return any(p.search(content) for p in LOW_VALUE_PATTERNS)


def normalize_answer(answer: str, chunks: List[CitationChunk]) -> tuple[str, Set[int]]:
    """
    Keep only well-cited bullets and enforce:
    - max 5 bullets
    - bullets must include at least one valid [n] citation
    - cited source section only contains referenced chunks
    """
    max_chunk = len(chunks)
    kept_bullets: List[str] = []
    used_ids: Set[int] = set()

    for bullet in extract_bullets(answer):
        cited_ids = {idx for idx in extract_citation_ids(bullet) if 1 <= idx <= max_chunk}
        if not cited_ids:
            continue
        if is_low_value_bullet(bullet):
            continue
        kept_bullets.append(bullet)
        used_ids.update(cited_ids)
        if len(kept_bullets) >= 5:
            break

    if not kept_bullets:
        return "Answer:\n- I don't know based on the provided documents.", set()

    cited_lines = [f"- [{i}] {chunks[i - 1].source_file} p.{chunks[i - 1].page_number}" for i in sorted(used_ids)]

    output = "Answer:\n" + "\n".join(kept_bullets)
    if cited_lines:
        output += "\n\nCited sources:\n" + "\n".join(cited_lines)
    return output, used_ids
