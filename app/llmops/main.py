import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from decouple import config

_steps = [
    "get_documents",
    "etl_chromdb_pdf",
    "etl_chromdb_scanned_pdf",
    "chain_of_thought"
]

GEMINI_API_KEY = config("GOOGLE_API_KEY", cast=str)



# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "get_documents" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_documents"),
                "main",
                parameters={
                    "document_folder": config["etl"]["document_folder"],
                    "path_document_folder": config["etl"]["path_document_folder"],
                    "artifact_name": config["etl"]["input_artifact_name"],
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
        if "etl_chromdb_pdf" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "etl_chromdb_pdf"),
                "main",
                parameters={
                    "input_artifact": f'{config["etl"]["input_artifact_name"]}:latest',
                    "output_artifact": "chromdb.zip",
                    "output_type": "chromdb",
                    "output_description": "Documents in pdf to be read and stored in chromdb",
                    "embedding_model": config["etl"]["embedding_model"]
                },
            )
        if "etl_chromdb_scanned_pdf" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "etl_chromdb_scanned_pdf"),
                "main",
                parameters={
                    "input_artifact": f'{config["etl"]["input_artifact_name"]}:latest',
                    "output_artifact": "chromdb.zip",
                    "output_type": "chromdb",
                    "output_description": "Scanned Documents in pdf to be read and stored in chromdb",
                    "embedding_model": config["etl"]["embedding_model"]
                },
            )
        
        if "chain_of_thought" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "chain_of_thought"),
                "main",
                parameters={
                    "query": config["prompt_engineering"]["query"],
                    "input_chromadb_artifact": "chromdb.zip:latest",
                    "embedding_model": config["etl"]["embedding_model"],
                },
            )

if __name__ == "__main__":
    go()
