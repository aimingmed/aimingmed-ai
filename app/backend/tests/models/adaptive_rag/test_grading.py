import pytest
from models.adaptive_rag import grading

def test_grade_documents_class():
    doc = grading.GradeDocuments(binary_score='yes')
    assert doc.binary_score == 'yes'

def test_grade_hallucinations_class():
    doc = grading.GradeHallucinations(binary_score='no')
    assert doc.binary_score == 'no'

def test_grade_answer_class():
    doc = grading.GradeAnswer(binary_score='yes')
    assert doc.binary_score == 'yes'
