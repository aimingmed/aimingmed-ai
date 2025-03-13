import os
import logging
import argparse
import mlflow
import chromadb
import shutil
from decouple import config
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY", cast=str)
os.environ["DEEPSEEK_API_KEY"] = config("DEEPSEEK_API_KEY", cast=str)
os.environ["MOONSHOT_API_KEY"] = config("MOONSHOT_API_KEY", cast=str)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGSMITH_API_KEY"] = config("LANGSMITH_API_KEY", cast=str)
os.environ["LANGSMITH_TRACING"] = config("LANGSMITH_TRACING", cast=str)
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = config("LANGSMITH_PROJECT", cast=str)

def go(args):

    # start a new MLflow run
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("development").experiment_id, run_name="etl_chromdb_pdf"):
        existing_params = mlflow.get_run(mlflow.active_run().info.run_id).data.params
        if 'query' not in existing_params:
            mlflow.log_param('query', args.query)
        
        # Log parameters to MLflow
        mlflow.log_params({
            "input_chromadb_artifact": args.input_chromadb_artifact,
            "embedding_model": args.embedding_model,
            "chat_model_provider": args.chat_model_provider
        })


        logger.info("Downloading chromadb artifact")
        artifact_chromadb_local_path = mlflow.artifacts.download_artifacts(artifact_uri=args.input_chromadb_artifact)

        # unzip the artifact
        logger.info("Unzipping the artifact")
        shutil.unpack_archive(artifact_chromadb_local_path, "chroma_db")

        # Load data from ChromaDB
        db_folder = "chroma_db"
        db_path = os.path.join(os.getcwd(), db_folder)
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection_name = "rag-chroma"
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
            )
            
        elif args.chat_model_provider == "gemini":
            # Initialize Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
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

    parser.add_argument(
        "--chat_model_provider",
        type=str,
        default="gemini",
        help="Chat model provider"
    )

    args = parser.parse_args()
    
    go(args)