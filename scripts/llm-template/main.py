import os
from decouple import config

from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


# Get the BASE_URL from the environment variables
GOOGLE_API_KEY = config("GOOGLE_API_KEY", cast=str)
WANDB_API_KEY = config("WANDB_API_KEY", cast=str)
LANGSMITH_API_KEY = config("LANGSMITH_API_KEY", cast=str)
LANGCHAIN_PROJECT = config("LANGCHAIN_PROJECT", cast=str)

# Set environment variables
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

embeddings = VertexAIEmbeddings(model="text-embedding-004")

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)