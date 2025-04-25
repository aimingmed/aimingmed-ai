import pytest
from models.adaptive_rag import routing

def test_route_query_class():
    route = routing.RouteQuery(datasource="vectorstore")
    assert route.datasource == "vectorstore"
