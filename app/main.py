from fastapi import FastAPI

app = FastAPI(
    title="RL Agents API",
    description="Serves tabular Q-learning agents trained in the rl_agents project.",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Tell the caller the service is up."""
    return {"status": "ok"}
