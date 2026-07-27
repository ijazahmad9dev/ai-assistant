from pathlib import Path
from docling.chunking import HybridChunker
from langchain_docling.loader import DoclingLoader, ExportType
from app.src.config import DATA_PATH, EMBED_MODEL_ID
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import gc

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

def load_pdfs_from_directory():
    data_dir = Path(DATA_PATH)
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    chunker = HybridChunker(tokenizer=EMBED_MODEL_ID)

    docs = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")
        loader = DoclingLoader(
            file_path=str(pdf_path),
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=chunker,
        )
        docs.extend(loader.load())
        gc.collect() 

    print(f"Loaded {len(docs)} chunks.")
    print(f"First chunk content: {docs[0].page_content[:50]}...")  # Print first 100 characters of the first chunk
    return docs