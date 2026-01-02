# embeddings.py

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional

from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModel
)

# Device agnostic code
DEVICE = torch.device("cpu")

# Text Embedding Model(BGE)
TEXT_MODEL_NAME = "BAAI/bge-base-en-v1.5"

text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
text_model.eval()

# Image Embeddong Model (CLIP)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_model.eval()

# Text Embeddings
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Create dense text embeddings using BGE.
    - Uses CLS token (correct for BGE)
    - L2 normalized (required for cosine/IP search)
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    inputs = text_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = text_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()

def embed_text(text: str) -> np.ndarray:
    """Single-text wrapper."""
    return embed_texts([text])[0]

# Image Embeddings
def embed_image(pil_image: Image.Image) -> np.ndarray:
    """
    Create image embeddings using CLIP.
    """

    if pil_image is None:
        raise ValueError("Image is None")

    pil_image = pil_image.convert("RGB")

    inputs = clip_processor(
        images=pil_image,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        features = clip_model.get_image_features(**inputs)
        features = F.normalize(features, p=2, dim=1)

    return features[0].cpu().numpy()

# Table Embedding
def embed_table(table_text: str) -> Optional[np.ndarray]:
    """
    Embed tables by converting them into structured text.
    Rejects low-signal or malformed tables.
    """

    if not table_text:
        return None

    # Normalize whitespace
    lines = [
        line.strip()
        for line in table_text.splitlines()
        if line.strip()
    ]

    # Must have header + at least one row
    if len(lines) < 2:
        return None

    # Require explicit column structure
    if not any(("|" in line or "\t" in line) for line in lines):
        return None

    cleaned_text = "\n".join(lines)

    # Avoid pure numeric garbage
    alpha_chars = sum(c.isalpha() for c in cleaned_text)
    if alpha_chars / max(len(cleaned_text), 1) < 0.10:
        return None

    structured_representation = (
        "TABLE DATA\n"
        "Structured tabular information follows:\n"
        f"{cleaned_text}"
    )

    return embed_text(structured_representation)