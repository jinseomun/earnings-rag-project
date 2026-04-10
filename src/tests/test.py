import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
texts = ["passage: Apple reported strong revenue growth."]
emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
print(emb.shape)