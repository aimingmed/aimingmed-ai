#!/usr/bin/env python
import argparse
import logging
import os
import wandb
import shutil
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def combine_chromadb(chromadb_pdf_path, chromadb_scanned_pdf_path, output_path):
    """
    Combines two ChromaDB instances into a single ChromaDB.
    """

    # Load the ChromaDB instances
    chromadb_pdf_client = chromadb.PersistentClient(path=chromadb_pdf_path)
    chromadb_scanned_pdf_client = chromadb.PersistentClient(path=chromadb_scanned_pdf_path)

    # Get the collections
    collection_name = "rag_experiment"
    try:
        chromadb_pdf_collection = chromadb_pdf_client.get_collection(name=collection_name)
    except ValueError as e:
        raise ValueError(f"Collection '{collection_name}' not found in ChromaDB at '{chromadb_pdf_path}'. Ensure the etl_chromdb_pdf step was run successfully.") from e
    try:
        chromadb_scanned_pdf_collection = chromadb_scanned_pdf_client.get_collection(name=collection_name)
    except ValueError as e:
        raise ValueError(f"Collection '{collection_name}' not found in ChromaDB at '{chromadb_scanned_pdf_path}'. Ensure the etl_chromdb_scanned_pdf step was run successfully.") from e

    # Get all data from the collections
    chromadb_pdf_data = chromadb_pdf_collection.get(include=["documents", "metadatas", "embeddings"])
    chromadb_scanned_pdf_data = chromadb_scanned_pdf_collection.get(include=["documents", "metadatas", "embeddings"])

    # Create a new ChromaDB instance
    combined_chromadb_client = chromadb.PersistentClient(path=output_path)
    combined_chromadb_collection = combined_chromadb_client.create_collection(name=collection_name)

    # Add the data to the combined ChromaDB
    combined_chromadb_collection.add(
        documents=chromadb_pdf_data["documents"] + chromadb_scanned_pdf_data["documents"],
        metadatas=chromadb_pdf_data["metadatas"] + chromadb_scanned_pdf_data["metadatas"],
        ids=chromadb_pdf_data["ids"] + chromadb_scanned_pdf_data["ids"],
        embeddings=chromadb_pdf_data["embeddings"] + chromadb_scanned_pdf_data["embeddings"],
    )

    logger.info(f"Combined ChromaDB created at {output_path}")


def go(args):
    """
    Run the combine chromadb component.
    """

    run = wandb.init(job_type="combine_chromadb", entity='aimingmed')
    run.config.update(args)

    # Download the ChromaDB artifacts
    logger.info("Downloading chromadb_pdf artifact")
    chromadb_pdf_artifact = run.use_artifact(args.chromadb_pdf_artifact).file()
    chromadb_pdf_path = os.path.join(chromadb_pdf_artifact, "chroma_db")

    logger.info("Downloading chromadb_scanned_pdf artifact")
    chromadb_scanned_pdf_artifact = run.use_artifact(args.chromadb_scanned_pdf_artifact).file()
    chromadb_scanned_pdf_path = os.path.join(chromadb_scanned_pdf_artifact, "chroma_db")

    # Create the output directory
    output_folder = "combined_chromadb"
    output_path = os.path.join(os.getcwd(), output_folder)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Combine the ChromaDB instances
    combine_chromadb(chromadb_pdf_path, chromadb_scanned_pdf_path, output_path)

    # Create a new artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    # Zip the database folder first
    shutil.make_archive(output_path, 'zip', output_path)

    # Add the database to the artifact
    artifact.add_file(output_path + '.zip')

    # Log the artifact
    run.log_artifact(artifact)

    # Finish the run
    run.finish()

    # clean up - remove zip
    os.remove(output_path + '.zip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two ChromaDB instances into one.")

    parser.add_argument(
        "--chromadb_pdf_artifact",
        type=str,
        required=True,
        help="Fully-qualified name for the ChromaDB PDF artifact",
    )
    parser.add_argument(
        "--chromadb_scanned_pdf_artifact",
        type=str,
        required=True,
        help="Fully-qualified name for the ChromaDB Scanned PDF artifact",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name for the output artifact",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type for the output artifact",
    )
    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Description for the output artifact",
    )

    args = parser.parse_args()
    go(args)