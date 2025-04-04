#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import glob
import json
import logging
import os
import mlflow
import shutil

import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


        # Initialize embedding model (do this ONCE)
        model_embedding = HuggingFaceEmbeddings(model_name=args.embedding_model)  # Or a multilingual model


        # Create database, delete the database directory if it exists
        db_folder = "chroma_db"
        db_path = os.path.join(os.getcwd(), db_folder)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path)


        logger.info("Downloading artifact")
        artifact_local_path = mlflow.artifacts.download_artifacts(artifact_uri=args.input_artifact)
        
        logger.info("Reading data")

        # unzip the downloaded artifact
        import zipfile
        with zipfile.ZipFile(artifact_local_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        # show the unzipped folder
        documents_folder = os.path.splitext(os.path.basename(artifact_local_path))[0]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=15000, chunk_overlap=7500
        )

        # read the dictionary json for word replacement in the read text
        with open(f'./{documents_folder}/2023CACA/CACA英文缩写.json', 'r', encoding='utf-8') as f:
            df_dict_json = json.load(f)

        ls_docs = []
        pdf_files = glob.glob(f"./{documents_folder}/**/*.pdf", recursive=True)
        
        for pdf_file in pdf_files:
            read_text = extract_chinese_text_from_pdf(pdf_file)
            relative_path = os.path.relpath(pdf_file, start=f"./{documents_folder}")

            # if the parent directory of the pdf file is 2023CACA, then replace the shortform text with the dictionary value
            if '2023CACA' in relative_path:
                # get the pdf filename without the extension
                pdf_filename = os.path.splitext(os.path.basename(pdf_file))[0]
                # replace the text with the dictionary
                dict_file = df_dict_json.get(pdf_filename)
                if dict_file:
                    for key, value in dict_file.items():
                        read_text = read_text.replace(key, value)
            

            document = Document(metadata={"file": relative_path}, page_content=read_text)
            ls_docs.append(document)


                                        
        doc_splits = text_splitter.split_documents(ls_docs)

        # Add to vectorDB
        _vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=model_embedding,
            persist_directory=db_path
        )

        logger.info("Logging artifact with mlflow")
        shutil.make_archive(db_path, 'zip', db_path)
        mlflow.log_artifact(db_path + '.zip', args.output_artifact)

        # clean up
        os.remove(db_path + '.zip')
        shutil.rmtree(db_path)
        shutil.rmtree(documents_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ETL for ChromaDB with readable PDF")

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