import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

_steps = [
    "get_documents",
    "etl_chromadb_pdf",
    "etl_chromadb_scanned_pdf", # the performance for scanned pdf may not be good
    "rag_cot_evaluation",
    "rag_adaptive_evaluation",
    "test_rag_cot"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the MLflow experiment. All runs will be grouped under this name
    mlflow.set_experiment(config["main"]["experiment_name"])

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory():

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
        if "etl_chromadb_pdf" in active_steps:
            if config["etl"]["run_id_documents"] == "None":
                # Look for run_id that has artifact logged as documents
                run_id = None
                client = mlflow.tracking.MlflowClient()
                for run in client.search_runs(experiment_ids=[client.get_experiment_by_name(config["main"]["experiment_name"]).experiment_id]):
                    for artifact in client.list_artifacts(run.info.run_id):
                        if artifact.path == "documents":
                            run_id = run.info.run_id
                            break
                    if run_id:
                        break

                if run_id is None:
                    raise ValueError("No run_id found with artifact logged as documents")
            else:
                run_id = config["etl"]["run_id_documents"]


            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "etl_chromadb_pdf"),
                "main",
                parameters={
                    "input_artifact": f'runs:/{run_id}/documents/documents.zip',
                    "output_artifact": "chromadb",
                    "output_type": "chromadb",
                    "output_description": "Documents in pdf to be read and stored in chromdb",
                    "embedding_model": config["etl"]["embedding_model"]
                },
            )

        if "etl_chromadb_scanned_pdf" in active_steps:

            if config["etl"]["run_id_documents"] == "None":
                # Look for run_id that has artifact logged as documents
                run_id = None
                client = mlflow.tracking.MlflowClient()
                for run in client.search_runs(experiment_ids=[client.get_experiment_by_name(config["main"]["experiment_name"]).experiment_id]):
                    for artifact in client.list_artifacts(run.info.run_id):
                        if artifact.path == "documents":
                            run_id = run.info.run_id
                            break
                    if run_id:
                        break

                if run_id is None:
                    raise ValueError("No run_id found with artifact logged as documents")
            else:
                run_id = config["etl"]["run_id_documents"]

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "etl_chromadb_scanned_pdf"),
                "main",
                parameters={
                    "input_artifact": f'runs:/{run_id}/documents/documents.zip',
                    "output_artifact": "chromadb",
                    "output_type": "chromadb",
                    "output_description": "Scanned Documents in pdf to be read and stored in chromdb",
                    "embedding_model": config["etl"]["embedding_model"]
                },
            )
        if "rag_cot_evaluation" in active_steps:

            if config["rag"]["run_id_chromadb"] == "None":
                # Look for run_id that has artifact logged as documents
                run_id = None
                client = mlflow.tracking.MlflowClient()
                for run in client.search_runs(experiment_ids=[client.get_experiment_by_name(config["main"]["experiment_name"]).experiment_id]):
                    for artifact in client.list_artifacts(run.info.run_id):
                        if artifact.path == "chromadb":
                            run_id = run.info.run_id
                            break
                    if run_id:
                        break

                if run_id is None:
                    raise ValueError("No run_id found with artifact logged as documents")
            else:
                run_id = config["rag"]["run_id_chromadb"]

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "rag_cot_evaluation"),
                "main",
                parameters={
                    "query": config["testing"]["query"],
                    "input_chromadb_artifact": f'runs:/{run_id}/chromadb/chroma_db.zip',
                    "embedding_model": config["etl"]["embedding_model"],
                    "chat_model_provider": config["rag"]["chat_model_provider"]
                },
            )
        
        if "rag_adaptive_evaluation" in active_steps:

            if config["rag"]["run_id_chromadb"] == "None":
                # Look for run_id that has artifact logged as documents
                run_id = None
                client = mlflow.tracking.MlflowClient()
                for run in client.search_runs(experiment_ids=[client.get_experiment_by_name(config["main"]["experiment_name"]).experiment_id]):
                    for artifact in client.list_artifacts(run.info.run_id):
                        if artifact.path == "chromadb":
                            run_id = run.info.run_id
                            break
                    if run_id:
                        break

                if run_id is None:
                    raise ValueError("No run_id found with artifact logged as documents")
            else:
                run_id = config["rag"]["run_id_chromadb"]

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "rag_adaptive_evaluation"),
                "main",
                parameters={
                    "query": config["testing"]["query"],
                    "evaluation_dataset_csv_path": config["evaluation"]["evaluation_dataset_csv_path"],
                    "evaluation_dataset_column_question": config["evaluation"]["evaluation_dataset_column_question"],
                    "evaluation_dataset_column_answer": config["evaluation"]["evaluation_dataset_column_answer"],
                    "input_chromadb_artifact": f'runs:/{run_id}/chromadb/chroma_db.zip',
                    "embedding_model": config["etl"]["embedding_model"],
                    "chat_model_provider": config["rag"]["chat_model_provider"],
                    "ls_chat_model_evaluator": ','.join(config["evaluation"]["ls_chat_model_provider"]) if config["evaluation"]["ls_chat_model_provider"] is not None else 'None',
                },
            )

        if "test_rag_cot" in active_steps:

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_rag_cot"),
                "main",
                parameters={
                    "query": config["testing"]["query"],
                    "input_chromadb_local": os.path.join(hydra.utils.get_original_cwd(), "src", "rag_cot_evaluation", "chroma_db"),
                    "embedding_model": config["etl"]["embedding_model"],
                    "chat_model_provider": config["rag"]["chat_model_provider"]
                },
            )


if __name__ == "__main__":
    go()
