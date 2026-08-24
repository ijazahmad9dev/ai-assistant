from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.src.config import DATA_PATH

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def _extract_text_from_pdf(pdf_path: Path):
    """Returns a list of (page_number, text) tuples for a single PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append((page_num + 1, text))  # 1-indexed page numbers
    doc.close()
    return pages


def load_pdfs_from_directory():
    data_dir = Path(DATA_PATH)
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_docs = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")
        pages = _extract_text_from_pdf(pdf_path)

        for page_num, page_text in pages:
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                all_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                        },
                    )
                )

    print(f"Loaded {len(all_docs)} chunks.")
    return all_docs