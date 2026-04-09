# test_embed_cpu.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
vec = model.encode(
    ["passage: hello world"],
    batch_size=1,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)
print(vec.shape)