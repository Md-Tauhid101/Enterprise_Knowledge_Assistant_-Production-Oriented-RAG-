# load.py
import os
import uuid
import hashlib
from typing import List
from unstructured.partition.auto import partition
from langchain_core.documents import Document

IMAGE_DIR = "./data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)


def compute_checksum(file_path: str) -> str:
    """
    Compute checksum to ensure document uniqueness.
    
    Args:
        file_path: List of input document path.
        
    Returns:
        A unique hash function of 256 bit.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load files into normalized Document objects WITHOUT embedding or chunking.
    This step produces raw structural elements, not final chunks.
    
    Args:
        file_path: List of document paths.
    
    Return:
        A list of normalized documents.
    """
    docs: List[Document] = []

    for file_path in file_paths:
        try:
            elements = partition(filename=file_path)
        except Exception as e:
            # fail-soft: skip bad files
            print(f"[WARN] Failed to parse {file_path}: {e}")
            continue

        file_extension = os.path.splitext(file_path)[-1].lower()
        doc_id = uuid.uuid4().hex

        checksum = compute_checksum(file_path=file_path)

        for idx, el in enumerate(elements):
            raw_type = type(el).__name__
            element_id = f"{doc_id}_{idx}"

            # IMAGE ELEMENT
            if raw_type == "Image" and hasattr(el, "image") and el.image is not None:
                image_name = f"{uuid.uuid4().hex}.png"
                image_path = os.path.join(IMAGE_DIR, image_name)

                # normalize image before saving
                el.image.convert("RGB").save(image_path, format="PNG")

                docs.append(
                    Document(
                        page_content="[IMAGE]",
                        metadata={
                            "doc_id": doc_id,
                            "element_id": element_id,
                            "source_path": file_path,
                            "file_ext": file_extension,
                            "element_type": "Image",
                            "image_path": image_path,
                        },
                    )
                )
                continue

            # TEXT / TABLE ELEMENT
            text = getattr(el, "text", None)
            if not text or not text.strip():
                continue

            raw_meta = el.metadata.to_dict() if el.metadata else {}

            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "doc_id": doc_id,
                        "element_id": element_id,
                        "source_path": file_path,
                        "file_ext": file_extension,
                        "source_type": file_extension,
                        "element_type": raw_type,  # keep truth
                        "page_number": raw_meta.get("page_number"),
                        "checksum": checksum,
                    },
                )
            )

    return docs

if __name__ == "__main__":
    docs = load_documents(["./data/India Post.pdf", "./data/instagram data.csv", "./data/ppt.pptx", "./data/Tauhid_CV.pdf"])

    print(f"Total documents loaded: {len(docs)}\n")

    # sanity check first 5 elements
    for d in docs[-10:]:
        print({
            "element_type": d.metadata.get("element_type"),
            "file": d.metadata.get("source"),
            "text_preview": d.page_content[:80]
        })

    from collections import Counter

    print("\nElement distribution:")
    print(Counter(d.metadata["element_type"] for d in docs))

    print("\nImage paths:")
    for d in docs:
        if d.metadata["element_type"] == "Image":
            print(d.metadata["image_path"])
            break