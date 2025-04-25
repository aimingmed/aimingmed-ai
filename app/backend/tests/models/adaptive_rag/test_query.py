import pytest
from models.adaptive_rag import query

def test_query_request_and_response():
    req = query.QueryRequest(query="What is AI?")
    assert req.query == "What is AI?"
    resp = query.QueryResponse(response="Artificial Intelligence")
    assert resp.response == "Artificial Intelligence"
