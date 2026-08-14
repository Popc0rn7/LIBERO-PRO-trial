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
    ):
        if execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")
        self.client = client
        self.execute_horizon = execute_horizon
        self.action_spec = action_spec or ActionSpec()
        self._queue: Deque[np.ndarray] = deque()
        self._episode_id = ""
        self._instruction = ""
        self.query_count = 0
        self.round_trip_latency_ms = []
        self.server_inference_latency_ms = []

    def reset(self, episode_id: str, instruction: str) -> None:
        self._queue.clear()
        self._episode_id = episode_id
        self._instruction = instruction
        self.query_count = 0
        self.round_trip_latency_ms = []
        self.server_inference_latency_ms = []
        self.client.reset(episode_id, instruction)

    def act(self, obs, step: int) -> np.ndarray:
        if not self._episode_id:
            raise RuntimeError("reset must be called before act")
        if not self._queue:
            request = PolicyRequest(
                instruction=self._instruction,
                observation=RawObservation.from_libero(obs),
            )
            import time
            started = time.monotonic()
            response = self.client.infer(request).validate(self.action_spec)
            self.round_trip_latency_ms.append((time.monotonic() - started) * 1000.0)
            self.query_count += 1
            latency = response.metadata.get("server_inference_latency_ms", response.metadata.get("inference_ms"))
            if latency is not None:
                self.server_inference_latency_ms.append(float(latency))
            for action in response.actions[: self.execute_horizon]:
                self._queue.append(action.copy())
        return self._queue.popleft()
