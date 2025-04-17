from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the model")


class QueryResponse(BaseModel):
    response: str = Field(..., description="The model's response")
