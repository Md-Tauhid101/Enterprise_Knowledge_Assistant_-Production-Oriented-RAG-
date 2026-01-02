# clean.py
import re
import unicodedata

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize unicode (fix weird spaces, quotes, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove page numbers like "Page 3 of 12"
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Fix hyphenated line breaks: informa-\ntion → information
    text = re.sub(r"-\s*\n\s*", "", text)

    # Merge line-wrapped sentences: newline between words → space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize bullet characters
    text = re.sub(r"[•–—]", "-", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
