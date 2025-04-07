from pydantic import BaseModel

class final_answer(BaseModel):
    """Final answer to be returned to the user."""
    answer: str