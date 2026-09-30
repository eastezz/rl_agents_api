"""Load trained Q-learning agents and pick their actions."""

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ACTION_NAMES = {
    "FrozenLake-v1": ["left", "down", "right", "up"],
    "Taxi-v4": ["south", "north", "east", "west", "pickup", "dropoff"],
}


class Agent:
    """A trained agent: its Q-table and the names of its actions."""

    def __init__(self, env_id, q_table, eval_rate):
        self.env_id = env_id
        self.q_table = q_table
        self.eval_rate = eval_rate
        self.n_states, self.n_actions = q_table.shape
        self.action_names = ACTION_NAMES[env_id]

    def best_action(self, state):
        """Return the action with the highest Q-value in this state."""
        if not 0 <= state < self.n_states:
            raise ValueError(f"state must be between 0 and {self.n_states - 1}, got {state}")
        return int(np.argmax(self.q_table[state]))


def load_agent(path):
    """Load one agent from an .npz file saved by rl_agents.train."""
    path = Path(path)
    with np.load(path) as data:
        return Agent(path.stem, data["q_table"], float(data["eval_rate"]))


def load_agents(models_dir=MODELS_DIR):
    """Load every .npz file in models_dir, keyed by environment id (the file name)."""
    return {path.stem: load_agent(path) for path in sorted(Path(models_dir).glob("*.npz"))}
