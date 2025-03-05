import os
import logging
import argparse
import mlflow
import chromadb
from decouple import config
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
GEMINI_API_KEY = config("GOOGLE_API_KEY", cast=str)
DEEKSEEK_API_KEY = config("DEEKSEEK_API_KEY", cast=str)
MOONSHOT_API_KEY = config("MOONSHOT_API_KEY", cast=str)

def stream_output(text):
    for char in text:
        print(char, end="")
        sys.stdout.flush()

def go(args):

    # start a new MLflow run
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("development").experiment_id, run_name="etl_chromadb_pdf"):
        existing_params = mlflow.get_run(mlflow.active_run().info.run_id).data.params
        if 'query' not in existing_params:
            mlflow.log_param('query', args.query)
        
        # Log parameters to MLflow
        mlflow.log_params({
            "input_chromadb_local": args.input_chromadb_local,
            "embedding_model": args.embedding_model,
            "chat_model_provider": args.chat_model_provider
        })


        # Load data from ChromaDB
        db_path = args.input_chromadb_local
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection_name = "rag_experiment"
        collection = chroma_client.get_collection(name=collection_name)

        # Formulate a question
        question = args.query

        if args.chat_model_provider == "deepseek":
            # Initialize DeepSeek model
            llm = ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=DEEKSEEK_API_KEY
            )
            
        elif args.chat_model_provider == "gemini":
            # Initialize Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=GEMINI_API_KEY,
                temperature=0,
                max_retries=3
                )
            
        elif args.chat_model_provider == "moonshot":
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

        # Initialize embedding model (do this ONCE)
        model = SentenceTransformer(args.embedding_model) 

        # Query (prompt)
        query_embedding = model.encode(question)  # Embed the query using the SAME model

        # Search ChromaDB
        documents_text = collection.query(query_embeddings=[query_embedding], n_results=5)

        # Generate chain of thought
        cot_output = cot_chain.invoke({"documents_text": documents_text, "question": question})
        print("Chain of Thought: ", end="")
        stream_output(cot_output.content)
        print()

        # Answer Prompt
        answer_template = """Given the chain of thought: {cot}
        Provide a concise answer to the question: {question}
        Provide the answer with language that is similar to the question asked.
        """
        answer_prompt = PromptTemplate(template=answer_template, input_variables=["cot", "question"])
        answer_chain = answer_prompt | llm

        # Generate answer
        answer_output = answer_chain.invoke({"cot": cot_output, "question": question})
        print("Answer: ", end="")
        stream_output(answer_output.content)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain of Thought RAG")

    parser.add_argument(
        "--query", 
        type=str,
        help="Question to ask the model",
        required=True
    )

    parser.add_argument(
        "--input_chromadb_local", 
        type=str,
        help="Path to input chromadb local directory",
        required=True
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Sentence Transformer model name"
    )

    parser.add_argument(
        "--chat_model_provider",
        type=str,
        default="gemini",
        help="Chat model provider"
    )

    args = parser.parse_args()
    
    go(args)