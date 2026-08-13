"""Fair, evaluator-owned action chunk execution."""

from collections import deque
from typing import Deque, Optional

import numpy as np

from .policy_client import PolicyClient
from .protocol import ActionSpec, PolicyRequest, RawObservation


class ActionChunkExecutor:
    """Consumes policy chunks using one shared replanning rule."""

    def __init__(
        self,
        client: PolicyClient,
        execute_horizon: int = 8,
        action_spec: Optional[ActionSpec] = None,
        clip_actions: bool = True,
    ):
        if execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")
        self.client = client
        self.execute_horizon = execute_horizon
        self.action_spec = action_spec or ActionSpec()
        self.clip_actions = clip_actions
        self._queue: Deque[np.ndarray] = deque()
        self._episode_id = ""
        self._instruction = ""

    def reset(self, episode_id: str, instruction: str) -> None:
        self._queue.clear()
        self._episode_id = episode_id
        self._instruction = instruction
        self.client.reset(episode_id, instruction)

    def act(self, obs, step: int) -> np.ndarray:
        if not self._episode_id:
            raise RuntimeError("reset must be called before act")
        if not self._queue:
            request = PolicyRequest(
                episode_id=self._episode_id,
                step=step,
                instruction=self._instruction,
                observation=RawObservation.from_libero(obs),
                action_spec=self.action_spec,
            )
            response = self.client.infer(request).validate(self.action_spec)
            for action in response.actions[: self.execute_horizon]:
                self._queue.append(action.copy())
        action = self._queue.popleft()
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        return action
