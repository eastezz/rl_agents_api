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


class EpisodeRequest(BaseModel):
    """Which game to play, and an optional seed to make the game repeatable."""

    env: str = Field(examples=["FrozenLake-v1"])
    seed: int | None = Field(default=None, ge=0, examples=[42])


class Step(BaseModel):
    """One move: where the agent was, what it did, and what happened."""

    state: int
    action: int
    action_name: str
    reward: float
    next_state: int


class EpisodeResponse(BaseModel):
    """The whole game: every move, the total reward, and how it ended."""

    env: str
    seed: int | None
    steps: list[Step]
    total_reward: float
    terminated: bool
    truncated: bool
