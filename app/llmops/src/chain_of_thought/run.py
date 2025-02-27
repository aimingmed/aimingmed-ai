import os
import logging
import argparse
import wandb
import chromadb
import shutil
from decouple import config
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GEMINI_API_KEY = config("GOOGLE_API_KEY", cast=str)


def go(args):
    run = wandb.init(job_type="chain_of_thought", entity='aimingmed')
    run.config.update(args)

    logger.info("Downloading chromadb artifact")
    artifact_chromadb_local_path = run.use_artifact(args.input_chromadb_artifact).file()

    # unzip the artifact
    logger.info("Unzipping the artifact")
    shutil.unpack_archive(artifact_chromadb_local_path, "chroma_db")

    # Load data from ChromaDB
    db_folder = "chroma_db"
    db_path = os.path.join(os.getcwd(), db_folder)
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection_name = "rag_experiment"
    collection = chroma_client.get_collection(name=collection_name)

    # Formulate a question
    question = args.query

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    

    # Chain of Thought Prompt
    cot_template = """Let's think step by step. 
    Given the following document in text: {documents_text}
    Question: {question}
    """
    cot_prompt = PromptTemplate(template=cot_template, input_variables=["documents_text", "question"])
    cot_chain = cot_prompt | llm

    # Initialize embedding model (do this ONCE)
    model = SentenceTransformer(args.embedding_model) 

    # Query (prompt)
    query_embedding = model.encode(question)  # Embed the query using the SAME model

    # Search ChromaDB
    documents_text = collection.query(query_embeddings=[query_embedding], n_results=5)

    # Generate chain of thought
    cot_output = cot_chain.invoke({"documents_text": documents_text, "question": question})
    print("Chain of Thought: ", cot_output)

    # Answer Prompt
    answer_template = """Given the chain of thought: {cot}
    Provide a concise answer to the question: {question}
    Provide the answer with language that is similar to the question asked.
    """
    answer_prompt = PromptTemplate(template=answer_template, input_variables=["cot", "question"])
    answer_chain = answer_prompt | llm

    # Generate answer
    answer_output = answer_chain.invoke({"cot": cot_output, "question": question})
    print("Answer: ", answer_output)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain of Thought RAG")

    parser.add_argument(
        "--query", 
        type=str,
        help="Question to ask the model",
        required=True
    )

    parser.add_argument(
        "--input_chromadb_artifact", 
        type=str,
        help="Fully-qualified name for the chromadb artifact",
        required=True
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Sentence Transformer model name"
    )

    args = parser.parse_args()
    
    go(args)