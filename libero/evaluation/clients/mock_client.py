"""Deterministic client for smoke tests and CI."""

import numpy as np

from ..policy_client import PolicyClient
from ..protocol import ActionSpec, PolicyResponse
from .registry import register_client


@register_client("mock")
class MockClient(PolicyClient):
    def __init__(self, action_spec, chunk_size=16, value=0.0):
        self.action_spec = action_spec
        self.chunk_size = int(chunk_size)
        self.value = float(value)
        self.inference_count = 0
        self.closed = False

    @classmethod
    def from_config(cls, cfg):
        action = dict(cfg.get("action", {}))
        spec = ActionSpec(**action)
        options = cfg.get("mock", {})
        return cls(spec, options.get("chunk_size", 16), options.get("value", 0.0))

    def reset(self, episode_id, instruction):
        self.episode_id = episode_id

    def infer(self, request):
        self.inference_count += 1
        actions = np.full((self.chunk_size, self.action_spec.dim), self.value, np.float32)
        return PolicyResponse(actions, self.action_spec)

    def close(self):
        self.closed = True
