import json
import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalize text while preserving tables and paragraph structure."""

    # Remove the infobox marker only (keep the infobox content)
    text = re.sub(r"\[Infobox.*?\]\n?", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters EXCEPT newline and tab
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)

    # Replace non-breaking spaces
    text = text.replace("\u00A0", " ")

    # Collapse only horizontal whitespace (spaces + tabs)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove trailing spaces on each line
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines (keep one empty line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_corpus(jsonl_path: str) -> list[Document]:
    """Load JSONL corpus into LangChain Documents."""

    documents = []
    seen_docs = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            doc_id = record["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            title = record["title"].strip()
            text = clean_text(record["text"])

            # Skip empty documents
            if len(text) < 30:
                continue

            page_content = f"Title: {title}\n\n{text}"

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "doc_id": doc_id,
                        "title": title,
                        "url": record["url"],
                        "wikidata_qid": record["wikidata_qid"],
                        "approx_tokens": record.get("approx_tokens"),
                    },
                )
            )

    return documents