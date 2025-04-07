from typing import List

from fastapi import APIRouter

from models.adaptive_rag.router import RouteQuery
from models.adaptive_rag.grading import GradeAnswer, GradeDocuments, GradeHallucinations

router = APIRouter()

@router.post("/", response_model=SummaryResponseSchema, status_code=201)
