import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from docx import Document
from pypdf import PdfReader


DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_documents"
META_FILE = Path("collection_meta.json")


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {page_num}]\n{text}")

    return "\n\n".join(pages)


def read_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return read_txt(file_path)
    if suffix == ".pdf":
        return read_pdf(file_path)
    if suffix == ".docx":
        return read_docx(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def save_collection_metadata(file_records):
    data = {
        "files": file_records,
        "file_count": len(file_records),
    }
    META_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    if not DATA_DIR.exists():
        print("The data folder does not exist.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    existing = collection.get()
    existing_ids = existing.get("ids", [])
    if existing_ids:
        print("Warning: collection already contains data.")
        print("If you want a clean rebuild, delete the chroma_db folder first.")

    file_records = []
    doc_count = 0
    chunk_count = 0

    for file_path in DATA_DIR.glob("**/*"):
        if file_path.is_dir():
            continue

        try:
            text = load_file(file_path)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        chunks = chunk_text(text)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{file_path.name}-{i}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_index": i,
                }
            )

        if documents:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            file_records.append(
                {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "chunks": len(documents),
                    "suffix": file_path.suffix.lower(),
                }
            )

            doc_count += 1
            chunk_count += len(documents)
            print(f"Indexed {file_path} with {len(documents)} chunks.")

    save_collection_metadata(file_records)

    print(f"Done. Indexed {doc_count} files and {chunk_count} chunks.")
    print("Saved file metadata to collection_meta.json")


if __name__ == "__main__":
    main()