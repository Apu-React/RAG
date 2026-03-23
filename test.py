# test_pdf.py
import fitz

doc = fitz.open(r"C:\Users\Admin\OneDrive\hr_rag\hr_policy_sample.pdf")
print(f"Pages: {len(doc)}")
for i, page in enumerate(doc):
    text = page.get_text()
    print(f"\n--- Page {i+1} ({len(text)} chars) ---")
    print(text[:300])