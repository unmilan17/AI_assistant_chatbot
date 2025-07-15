from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chatbot_ml_ai_papers import graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    user_input = request.message
    response = graph.invoke({"messages": ["user", user_input]}, config={"configurable": {"thread_id": "1"}})
    last_msg = response["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return {"response": content}
