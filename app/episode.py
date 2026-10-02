"""Let a trained agent play one full game in its gymnasium environment."""

import gymnasium as gym


def play_episode(agent, seed=None):
    """Play one game with the agent's best actions and record every step."""
    env = gym.make(agent.env_id)
    state, _ = env.reset(seed=seed)
    steps = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.best_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps.append(
            {
                "state": int(state),
                "action": action,
                "action_name": agent.action_names[action],
                "reward": float(reward),
                "next_state": int(next_state),
            }
        )
        state = next_state
    env.close()
    return steps, terminated, truncated
