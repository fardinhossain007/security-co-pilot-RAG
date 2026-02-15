import unittest

from app.postprocess import CitationChunk, extract_citation_ids, normalize_answer


def _chunks(n: int):
    return [
        CitationChunk(text=f"chunk {i}", source_file=f"doc{i}.pdf", page_number=i)
        for i in range(1, n + 1)
    ]


class TestRagNormalization(unittest.TestCase):
    def test_extract_citation_ids_variants(self):
        text = "A [1][2] B [3, 4] C"
        ids = extract_citation_ids(text)
        self.assertEqual(ids, {1, 2, 3, 4})

    def test_drops_low_value_source_listing_bullets(self):
        answer = """Answer:
- [1] NIST.CSWP.29.pdf | PAGE: 20
- The functions are meant to be addressed concurrently as an integrated cycle. [2]

Cited sources:
- [1] NIST.CSWP.29.pdf p.20
- [2] NIST.CSWP.29.pdf p.31
"""
        normalized, used_ids = normalize_answer(answer, _chunks(5))
        self.assertIn("integrated cycle", normalized)
        self.assertNotIn("| PAGE:", normalized)
        self.assertEqual(used_ids, {2})

    def test_enforces_max_five_bullets(self):
        bullets = "\n".join([f"- Point {i} [1]" for i in range(1, 9)])
        answer = f"Answer:\n{bullets}\n"
        normalized, _ = normalize_answer(answer, _chunks(3))
        normalized_bullets = [ln for ln in normalized.splitlines() if ln.strip().startswith("- ")]
        self.assertLessEqual(len(normalized_bullets), 6)  # includes cited sources bullets
        answer_bullets = []
        in_answer = False
        for ln in normalized.splitlines():
            if ln.strip() == "Answer:":
                in_answer = True
                continue
            if ln.strip() == "Cited sources:":
                break
            if in_answer and ln.strip().startswith("- "):
                answer_bullets.append(ln)
        self.assertEqual(len(answer_bullets), 5)

    def test_falls_back_when_no_valid_bullets(self):
        answer = """Answer:
- None used. [1]
- [2] NIST.CSWP.29.pdf | PAGE: 31
"""
        normalized, used_ids = normalize_answer(answer, _chunks(3))
        self.assertIn("I don't know based on the provided documents.", normalized)
        self.assertEqual(used_ids, set())


if __name__ == "__main__":
    unittest.main()
