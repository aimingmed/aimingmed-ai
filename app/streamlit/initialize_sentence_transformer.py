from decouple import config
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = config("EMBEDDING_MODEL", cast=str, default="paraphrase-multilingual-mpnet-base-v2")

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL) 