#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import mlflow
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    zip_path = os.path.join(args.path_document_folder, f"{args.document_folder}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', args.path_document_folder, args.document_folder)

    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("development").experiment_id):

        existing_params = mlflow.get_run(mlflow.active_run().info.run_id).data.params
        if 'artifact_description' not in existing_params:
            mlflow.log_param('artifact_description', args.artifact_description)
        if 'artifact_types' not in existing_params:
            mlflow.log_param('artifact_types', args.artifact_type)
        

        # Log parameters to MLflow
        mlflow.log_params({
            "input_artifact": args.artifact_name,
        })

        logger.info(f"Uploading {args.artifact_name} to MLFlow")
        mlflow.log_artifact(zip_path, args.artifact_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("document_folder", type=str, help="Name of the sample to download")

    parser.add_argument("path_document_folder", type=str, help="Path to the document folder")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)