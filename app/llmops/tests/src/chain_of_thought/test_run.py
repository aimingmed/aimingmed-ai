import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.append("/Users/leehongkai/projects/aimingmed/aimingmed-ai/app/llmops")
from src.chain_of_thought import run

def test_go():
    # Create mock arguments
    args = MagicMock()
    args.query = "test_query"
    args.input_chromadb_artifact = "test_artifact"
    args.embedding_model = "test_embedding_model"
    args.chat_model_provider = "gemini"

    # Mock wandb.init and other external dependencies
    with patch("wandb.init") as mock_wandb_init, \
         patch("chromadb.PersistentClient") as mock_chromadb_client, \
         patch("sentence_transformers.SentenceTransformer") as mock_sentence_transformer, \
         patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_chat_google_generative_ai:

        # Configure the mocks
        mock_wandb_init.return_value = MagicMock()
        mock_chromadb_client.return_value = MagicMock()
        mock_sentence_transformer.return_value = MagicMock()
        mock_chat_google_generative_ai.return_value = MagicMock()

        # Call the go function
        run.go(args)

        # Add assertions to validate the behavior of the go function
        assert mock_wandb_init.called
