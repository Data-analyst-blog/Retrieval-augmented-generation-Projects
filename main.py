from fastapi import FastAPI
from pydantic import BaseModel

from rag.chat import generate_answer
from state.session_manager import SessionManager

app = FastAPI()
session_manager = SessionManager()

class Query(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
def chat(query: Query):

    history = session_manager.get_history(query.session_id)

    result = generate_answer(query.question, history)

    # Extract only answer for session memory
    answer_text = result["answer"]

    session_manager.update_history(
        query.session_id,
        query.question,
        answer_text
    )

    return result