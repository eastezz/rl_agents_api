import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agents_lists_both_games():
    response = client.get("/agents")
    assert response.status_code == 200
    assert [agent["env"] for agent in response.json()] == ["FrozenLake-v1", "Taxi-v4"]


def test_predict_returns_best_action():
    response = client.post("/predict", json={"env": "FrozenLake-v1", "state": 0})
    assert response.status_code == 200
    body = response.json()
    assert len(body["q_values"]) == 4
    assert body["action"] == body["q_values"].index(max(body["q_values"]))
    assert body["action_name"] == ["left", "down", "right", "up"][body["action"]]


def test_predict_unknown_game_returns_404():
    response = client.post("/predict", json={"env": "Chess-v0", "state": 0})
    assert response.status_code == 404


@pytest.mark.parametrize("state", [-1, 16, "abc"])
def test_predict_invalid_state_returns_422(state):
    response = client.post("/predict", json={"env": "FrozenLake-v1", "state": state})
    assert response.status_code == 422


def test_predict_missing_state_returns_422():
    response = client.post("/predict", json={"env": "FrozenLake-v1"})
    assert response.status_code == 422


def test_episode_plays_until_the_game_ends():
    response = client.post("/episode", json={"env": "FrozenLake-v1", "seed": 42})
    assert response.status_code == 200
    body = response.json()
    assert body["terminated"] or body["truncated"]
    assert body["total_reward"] == sum(step["reward"] for step in body["steps"])
    for before, after in zip(body["steps"], body["steps"][1:]):
        assert after["state"] == before["next_state"]


def test_episode_with_the_same_seed_is_repeatable():
    first = client.post("/episode", json={"env": "Taxi-v4", "seed": 7}).json()
    second = client.post("/episode", json={"env": "Taxi-v4", "seed": 7}).json()
    assert first == second


def test_episode_uses_the_agents_best_actions():
    body = client.post("/episode", json={"env": "Taxi-v4", "seed": 42}).json()
    for step in body["steps"]:
        answer = client.post("/predict", json={"env": "Taxi-v4", "state": step["state"]}).json()
        assert step["action"] == answer["action"]


def test_episode_unknown_game_returns_404():
    response = client.post("/episode", json={"env": "Chess-v0"})
    assert response.status_code == 404


def test_episode_negative_seed_returns_422():
    response = client.post("/episode", json={"env": "FrozenLake-v1", "seed": -1})
    assert response.status_code == 422
