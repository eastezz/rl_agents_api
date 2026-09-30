"""Shapes of the JSON the API accepts and returns."""

from pydantic import BaseModel, Field


class AgentInfo(BaseModel):
    """What GET /agents says about one trained agent."""

    env: str
    n_states: int
    n_actions: int
    action_names: list[str]
    success_rate: float


class PredictRequest(BaseModel):
    """The question: which game, and which state the agent is in."""

    env: str = Field(examples=["FrozenLake-v1"])
    state: int = Field(ge=0, examples=[0])


class PredictResponse(BaseModel):
    """The answer: the best action, its name, and the Q-values behind it."""

    env: str
    state: int
    action: int
    action_name: str
    q_values: list[float]
