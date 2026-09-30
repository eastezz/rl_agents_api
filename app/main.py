from fastapi import FastAPI, HTTPException

from app.agents import load_agents
from app.schemas import AgentInfo, PredictRequest, PredictResponse

app = FastAPI(
    title="RL Agents API",
    description="Serves tabular Q-learning agents trained in the rl_agents project.",
    version="0.1.0",
)

# Loaded once, when the server starts, and shared by all requests.
AGENTS = load_agents()


@app.get("/health")
def health():
    """Tell the caller the service is up."""
    return {"status": "ok"}


@app.get("/agents")
def list_agents() -> list[AgentInfo]:
    """List the trained agents this API can serve."""
    return [
        AgentInfo(
            env=agent.env_id,
            n_states=agent.n_states,
            n_actions=agent.n_actions,
            action_names=agent.action_names,
            success_rate=agent.eval_rate,
        )
        for agent in AGENTS.values()
    ]


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    """Return the trained agent's best action for a state."""
    agent = AGENTS.get(request.env)
    if agent is None:
        available = ", ".join(AGENTS)
        raise HTTPException(
            status_code=404,
            detail=f"No agent for {request.env}. Available: {available}",
        )
    try:
        action = agent.best_action(request.state)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictResponse(
        env=request.env,
        state=request.state,
        action=action,
        action_name=agent.action_names[action],
        q_values=agent.q_table[request.state].tolist(),
    )
