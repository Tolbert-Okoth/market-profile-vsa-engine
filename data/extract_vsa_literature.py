"""
Exhaustive Extraction & Deep Analysis of Volume Spread Analysis (VSA) Literature.
Extracts all pages and concepts from:
1. mtm_251058.pdf (Master the Markets by Tom Williams)
2. 1st3Chapters_5_24.pdf
3. Beginners_guide.pdf
"""
import os
import json
import fitz  # PyMuPDF

pdf_files = [
    "data/mtm_251058.pdf",
    "data/1st3Chapters_5_24.pdf",
    "data/Beginners_guide.pdf"
]

extracted_corpus = {}

for pdf_path in pdf_files:
    if not os.path.exists(pdf_path):
        print(f"Warning: {pdf_path} not found!")
        continue
    
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    total_pages = len(doc)
    print(f"Reading {file_name}: {total_pages} pages...")
    
    pages_text = []
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        pages_text.append({
            "page": page_num + 1,
            "text": text
        })
    
    extracted_corpus[file_name] = {
        "total_pages": total_pages,
        "pages": pages_text
    }
    doc.close()

# Save complete raw extraction to JSON
out_json = "data/extracted_vsa_corpus.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(extracted_corpus, f, ensure_ascii=False, indent=2)

print(f"Complete extracted VSA corpus saved to {out_json} ({os.path.getsize(out_json):,} bytes)")
