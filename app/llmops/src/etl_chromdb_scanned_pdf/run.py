#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import wandb
import shutil

import chromadb
# from openai import OpenAI
from typing import List
import numpy as np
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

    run = wandb.init(job_type="etl_chromdb_scanned_pdf", entity='aimingmed')
    run.config.update(args)

    # Setup the Gemini client
    # client = OpenAI(
    #     api_key=args.gemini_api_key,
    #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    # )


    # def get_google_embedding(text: str) -> List[float]:
    #     response = client.embeddings.create(
    #         model="text-embedding-004",
    #         input=text
    #     )
    #     return response.data[0].embedding

    # class GeminiEmbeddingFunction(object):
    #     def __init__(self, api_key: str, base_url: str, model_name: str):
    #         self.client = OpenAI(
    #             api_key=args.gemini_api_key,
    #             base_url=base_url
    #         )
    #         self.model_name = model_name

    #     def __call__(self, input: List[str]) -> List[List[float]]:
    #             all_embeddings = []
    #             for text in input:
    #                 response = self.client.embeddings.create(input=text, model=self.model_name)
    #                 embeddings = [record.embedding for record in response.data]
    #                 all_embeddings.append(np.array(embeddings[0]))
    #             return all_embeddings


    # Initialize embedding model (do this ONCE)
    model_embedding = SentenceTransformer('all-mpnet-base-v2')  # Or a multilingual model


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
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading data")

    # unzip the downloaded artifact
    import zipfile
    with zipfile.ZipFile(artifact_local_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(artifact_local_path)

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