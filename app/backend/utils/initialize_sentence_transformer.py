from decouple import config
from sentence_transformers import SentenceTransformer
import os

EMBEDDING_MODEL = config("EMBEDDING_MODEL", cast=str, default="paraphrase-multilingual-mpnet-base-v2")

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

# create directory if not exists
if not os.path.exists("./transformer_model"):
    os.makedirs("./transformer_model")
    
# save the model
model.save("./transformer_model/paraphrase-multilingual-mpnet-base-v2")
