import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_OUTPUT_FILES = [
    (
        Path("data/processed/earnings_calls_train.json"),
        Path("data/processed/earnings_call_chunks_train.json"),
    ),
    (
        Path("data/processed/earnings_calls_test.json"),
        Path("data/processed/earnings_call_chunks_test.json"),
    ),
]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_chunk_id(metadata: dict, chunk_index: int) -> str:
    call_id = metadata.get("call_id") or "unknown_call"
    return f"{call_id}_chunk_{chunk_index}"


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_docs = []

    for doc_index, doc in enumerate(docs):
        text = doc.get("page_content", "")
        metadata = doc.get("metadata", {})

        if not text or not text.strip():
            continue

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(split_texts):
            new_metadata = {
                **metadata,
                "doc_index": doc_index,
                "chunk_index": chunk_index,
                "chunk_id": build_chunk_id(metadata, chunk_index),
            }

            chunked_docs.append(
                {
                    "page_content": chunk_text.strip(),
                    "metadata": new_metadata,
                }
            )

    return chunked_docs


def main():
    print("Starting chunking...")

    for input_path, output_path in INPUT_OUTPUT_FILES:
        print(f"\nLoading: {input_path}")
        docs = load_json(input_path)
        print(f"Loaded {len(docs)} documents")

        chunked_docs = chunk_documents(docs)
        save_json(chunked_docs, output_path)

        print(f"Saved {len(chunked_docs)} chunks to {output_path}")

        if chunked_docs:
            print("\nSample chunk metadata:")
            print(chunked_docs[0]["metadata"])

            print("\nSample chunk text:")
            print(chunked_docs[0]["page_content"][:500])


if __name__ == "__main__":
    main()