#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import mlflow
import shutil

import chromadb
# from openai import OpenAI
import pytesseract as pt
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_text_from_pdf_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path) # Convert PDF pages to images
        extracted_text = ""
        for image in images:
            text = pt.image_to_string(image, lang="chi_sim+eng")  # chi_sim for Simplified Chinese, chi_tra for Traditional

            extracted_text += text + "\n"
        return extracted_text

    except ImportError:
      print("Error: pdf2image or pytesseract not installed. Please install them: pip install pdf2image pytesseract")
      return ""
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""



def go(args):
    """
    Run the etl for chromdb with scanned pdf
    """

    # Start an MLflow run
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("development").experiment_id, run_name="etl_chromdb_pdf"):
        existing_params = mlflow.get_run(mlflow.active_run().info.run_id).data.params
        if 'output_description' not in existing_params:
            mlflow.log_param('output_description', args.output_description)

        # Log parameters to MLflow
        mlflow.log_params({
            "input_artifact": args.input_artifact,
            "output_artifact": args.output_artifact,
            "output_type": args.output_type,
            "embedding_model": args.embedding_model
        })


        # Initialize embedding model 
        model_embedding = SentenceTransformer(args.embedding_model)  # Or a multilingual model


        # Create database, delete the database directory if it exists
        db_folder = "chroma_db"
        db_path = os.path.join(os.getcwd(), db_folder)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path)

        chroma_client = chromadb.PersistentClient(path=db_path)
        collection_name = "rag_experiment"
        db = chroma_client.create_collection(name=collection_name)


        logger.info("Downloading artifact")
        artifact_local_path = mlflow.artifacts.download_artifacts(artifact_uri=args.input_artifact)

        logger.info("Reading data")

        # unzip the downloaded artifact
        import zipfile
        with zipfile.ZipFile(artifact_local_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        # show the unzipped folder
        documents_folder = os.path.splitext(os.path.basename(artifact_local_path))[0]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for root, _dir, files in os.walk(f"./{documents_folder}"):
            for file in files:
                if file.endswith(".pdf"):
                    read_text = extract_text_from_pdf_ocr(os.path.join(root, file))
                    document = Document(page_content=read_text)
                    all_splits = text_splitter.split_documents([document])
                    
                    for i, split in enumerate(all_splits):
                        db.add(documents=[split.page_content], 
                            metadatas=[{"filename": file}],
                            ids=[f'{file[:-4]}-{str(i)}'],
                            embeddings=[model_embedding.encode(split.page_content)]
                    )
                        
        logger.info("Uploading artifact to MLFlow")
        shutil.make_archive(db_path, 'zip', db_path)
        mlflow.log_artifact(db_path + '.zip', args.output_artifact)

        # clean up
        os.remove(db_path + '.zip')
        shutil.rmtree(db_path)
        shutil.rmtree(documents_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type for the artifact output",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the artifact",
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