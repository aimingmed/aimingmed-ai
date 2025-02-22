#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os


import wandb

from wandb_utils.log_artifact import log_artifact
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    zip_path = os.path.join(args.path_document_folder, f"{args.document_folder}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', args.path_document_folder, args.document_folder)

    run = wandb.init(job_type="get_documents", entity='aimingmed')
    run.config.update(args)

    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        zip_path,
        run,
    )


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