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
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_chinese_text_from_pdf(pdf_path):
    """
    Extracts Chinese text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted Chinese text, or None if an error occurs.
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    try:
        with open(pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

        return text

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        converter.close()
        fake_file_handle.close()


def go(args):
    """
    Run the etl for chromdb with scanned pdf
    """

    run = wandb.init(job_type="etl_chromdb_scanned_pdf", entity='aimingmed')
    run.config.update(args)


    # Initialize embedding model (do this ONCE)
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
                read_text = extract_chinese_text_from_pdf(os.path.join(root, file))
                document = Document(page_content=read_text)
                all_splits = text_splitter.split_documents([document])
                
                for i, split in enumerate(all_splits):
                    db.add(documents=[split.page_content], 
                        metadatas=[{"filename": file}],
                        ids=[f'{file[:-4]}-{str(i)}'],
                        embeddings=[model_embedding.encode(split.page_content)]
                    )

    # Create a new artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    # zip the database folder first
    shutil.make_archive(db_path, 'zip', db_path)

    # Add the database to the artifact
    artifact.add_file(db_path + '.zip')

    # Log the artifact
    run.log_artifact(artifact)

    # Finish the run
    run.finish()

    # clean up
    os.remove(db_path + '.zip')
    os.remove(db_path)


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