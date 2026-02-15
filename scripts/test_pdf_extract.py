
from pathlib import Path
from pypdf import PdfReader

RAW = Path("/Users/fardin/Documents/Personal ML Projects/security-copilot-rag/data/raw")

pdfs = sorted(RAW.glob("*.pdf"))
if not pdfs:
    raise SystemExit("No PDFs found in data/raw. Download and place PDFs there first.")

for pdf in pdfs:
    reader = PdfReader(str(pdf))
    pages = len(reader.pages)
    sample = ""
    for i in range(min(2, pages)):
        txt = reader.pages[i].extract_text() or ""
        sample += txt.strip()[:400] + "\n"

    print("=" * 80)
    print(f"{pdf.name} | pages={pages}")
    print(sample.strip()[:800] if sample.strip() else "[No extractable text found]")
