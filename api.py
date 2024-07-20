# src/interfaces/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.core.agent import Agent

app = FastAPI()

# Initialize the main agent
main_agent = Agent("MainAgent")

class TaskType(BaseModel):
    task_type: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Agent API!"}

@app.post("/execute_task")
async def execute_task(task: str, user: str):
    result = main_agent.execute_task(task, user)
    return {"result": result}

@app.post("/create_agent")
async def create_agent(task_type: TaskType):
    try:
        new_agent = main_agent.create_specialized_agent(task_type.task_type)
        return {"status": "success", "new_agent_id": new_agent.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    agents = [str(agent) for agent in [main_agent] + main_agent.child_agents]
    return {"agents": agents}

@app.get("/agent/{agent_id}")
async def get_agent(agent_id: str):
    if agent_id == main_agent.id:
        return {"agent": str(main_agent)}
    for agent in main_agent.child_agents:
        if agent.id == agent_id:
            return {"agent": str(agent)}
    raise HTTPException(status_code=404, detail="Agent not found")

@app.post("/process_text")
async def process_text(text: str):
    result = main_agent.process_text(text)
    return {"result": result}

@app.get("/memory")
async def get_memory():
    memory = main_agent.get_memory()
    return {"memory": memory}