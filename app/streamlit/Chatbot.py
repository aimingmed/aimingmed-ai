import os
import streamlit as st
import chromadb
from decouple import config
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GEMINI_API_KEY = config("GOOGLE_API_KEY", cast=str)
DEEKSEEK_API_KEY = config("DEEKSEEK_API_KEY", cast=str)
MOONSHOT_API_KEY = config("MOONSHOT_API_KEY", cast=str)
CHAT_MODEL_PROVIDER = config("CHAT_MODEL_PROVIDER", cast=str)
INPUT_CHROMADB_LOCAL = config("INPUT_CHROMADB_LOCAL", cast=str)
EMBEDDING_MODEL = config("EMBEDDING_MODEL", cast=str)
COLLECTION_NAME = config("COLLECTION_NAME", cast=str)

st.title("💬 RAG AI for Medical Guideline")
st.caption(f"🚀 A RAG AI for Medical Guideline powered by {CHAT_MODEL_PROVIDER}")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Load data from ChromaDB
chroma_client = chromadb.PersistentClient(path=INPUT_CHROMADB_LOCAL)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL) 

if CHAT_MODEL_PROVIDER == "deepseek":
    # Initialize DeepSeek model
    llm = ChatDeepSeek(
        model="deepseek-chat", 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=DEEKSEEK_API_KEY
    )
    
elif CHAT_MODEL_PROVIDER == "gemini":
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        max_retries=3
        )
    
elif CHAT_MODEL_PROVIDER == "moonshot":
    # Initialize Moonshot model
    llm = Moonshot(
        model="moonshot-v1-128k", 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=MOONSHOT_API_KEY
    )

# Chain of Thought Prompt
cot_template = """Let's think step by step. 
Given the following document in text: {documents_text}
Question: {question}
Reply with language that is similar to the language used with asked question.
"""
cot_prompt = PromptTemplate(template=cot_template, input_variables=["documents_text", "question"])
cot_chain = cot_prompt | llm

# Answer Prompt
answer_template = """Given the chain of thought: {cot}
Provide a concise answer to the question: {question}
Provide the answer with language that is similar to the question asked.
"""
answer_prompt = PromptTemplate(template=answer_template, input_variables=["cot", "question"])
answer_chain = answer_prompt | llm

if prompt := st.chat_input():
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Query (prompt)
    query_embedding = model.encode(prompt)  # Embed the query using the SAME model

    # Search ChromaDB
    documents_text = collection.query(query_embeddings=[query_embedding], n_results=5)

    # Generate chain of thought
    cot_output = cot_chain.invoke({"documents_text": documents_text, "question": prompt})

    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = cot_output.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # Generate answer
    answer_output = answer_chain.invoke({"cot": cot_output, "question": prompt})
    msg = answer_output.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)



