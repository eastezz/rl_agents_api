import numpy as np
import pytest

from app.agents import MODELS_DIR, load_agent, load_agents


def make_model(folder, env_id, q_table, eval_rate=0.5):
    """Save a small fake model the same way rl_agents.train does."""
    path = folder / f"{env_id}.npz"
    np.savez(path, q_table=np.array(q_table, dtype=float), eval_rate=eval_rate)
    return path


def test_best_action_is_highest_q_value(tmp_path):
    agent = load_agent(make_model(tmp_path, "FrozenLake-v1", [[0.1, 0.9, 0.3, 0.2], [0.5, 0.1, 0.2, 0.7]]))
    assert agent.best_action(0) == 1
    assert agent.best_action(1) == 3
    assert agent.action_names[agent.best_action(0)] == "down"


@pytest.mark.parametrize("state", [-1, 2])
def test_state_outside_table_raises(tmp_path, state):
    agent = load_agent(make_model(tmp_path, "FrozenLake-v1", [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        agent.best_action(state)


def test_load_agents_uses_file_names_as_env_ids(tmp_path):
    make_model(tmp_path, "FrozenLake-v1", np.zeros((16, 4)))
    make_model(tmp_path, "Taxi-v4", np.zeros((500, 6)))
    assert sorted(load_agents(tmp_path)) == ["FrozenLake-v1", "Taxi-v4"]


def test_real_models_load():
    agents = load_agents(MODELS_DIR)
    assert agents["FrozenLake-v1"].q_table.shape == (16, 4)
    assert agents["Taxi-v4"].q_table.shape == (500, 6)
