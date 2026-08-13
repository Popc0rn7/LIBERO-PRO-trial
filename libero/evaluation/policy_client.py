"""Policy client interface used by the shared evaluator."""

from abc import ABC, abstractmethod

from .protocol import PolicyRequest, PolicyResponse


class PolicyClient(ABC):
    """A stateful policy living locally or in another Python environment."""

    @abstractmethod
    def reset(self, episode_id: str, instruction: str) -> None:
        """Start an episode and clear all policy-side temporal state."""

    @abstractmethod
    def infer(self, request: PolicyRequest) -> PolicyResponse:
        """Return one or more actions for the current observation."""

    def close(self) -> None:
        """Release transport resources."""

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
