"""OpenPI MessagePack/NumPy WebSocket client."""

import time
from typing import Any, Dict

import numpy as np

from ..policy_client import ClientInfo, PolicyClient
from ..protocol import ActionSpec, PolicyRequest, PolicyResponse
from .registry import register_client


class PolicyConnectionError(RuntimeError):
    pass


class PolicyTimeoutError(TimeoutError):
    pass


@register_client("openpi")
class OpenPIClient(PolicyClient):
    def __init__(self, host: str, port: int, timeout_seconds: float = 30.0, action_chunk_size=None):
        if timeout_seconds <= 0:
            raise ValueError("connection.timeout_seconds must be positive")
        self.host, self.port = host, int(port)
        self.timeout_seconds = float(timeout_seconds)
        self.action_chunk_size = None if action_chunk_size is None else int(action_chunk_size)
        self._connection = None
        self._metadata: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg):
        connection = cfg.get("connection", {})
        if not connection.get("host") or connection.get("port") is None:
            raise ValueError("openpi connection.host and connection.port are required")
        inference = cfg.get("inference", {})
        return cls(connection["host"], connection["port"], connection.get("timeout_seconds", 30),
                   inference.get("action_chunk_size"))

    @staticmethod
    def _codec():
        try:
            from openpi_client import msgpack_numpy
        except ImportError as exc:
            raise ImportError("OpenPIClient requires openpi-client==0.1.2") from exc
        return msgpack_numpy

    def _connect(self):
        if self._connection is not None:
            return
        try:
            from websockets.sync.client import connect
            self._connection = connect(
                "ws://{}:{}".format(self.host, self.port),
                open_timeout=self.timeout_seconds,
                close_timeout=self.timeout_seconds,
            )
            raw = self._connection.recv(timeout=self.timeout_seconds)
            self._metadata = dict(self._codec().unpackb(raw))
        except TimeoutError as exc:
            self.close()
            raise PolicyTimeoutError("timed out connecting to policy server") from exc
        except Exception as exc:
            self.close()
            raise PolicyConnectionError("policy connection failed: {}".format(exc)) from exc

    def check(self):
        self._connect()
        return ClientInfo(True, "openpi", str(self._metadata.get("model_name", "")), dict(self._metadata))

    def close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def reset(self, episode_id: str, instruction: str) -> None:
        self._episode_id = episode_id
        self._instruction = instruction

    def infer(self, request: PolicyRequest) -> PolicyResponse:
        self._connect()
        payload = self._default_request_adapter(request)
        try:
            self._connection.send(self._codec().packb(payload))
            raw = self._connection.recv(timeout=self.timeout_seconds)
            if isinstance(raw, str):
                raise ValueError(raw)
            result = self._codec().unpackb(raw)
        except TimeoutError as exc:
            self.close()
            raise PolicyTimeoutError("policy inference timed out") from exc
        except (PolicyTimeoutError, ValueError):
            raise
        except Exception as exc:
            self.close()
            raise PolicyConnectionError("policy disconnected: {}".format(exc)) from exc
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError("OpenPI response does not contain 'actions'")
        spec_dict = result.get("action_spec", {})
        spec = ActionSpec(**spec_dict) if spec_dict else request.action_spec
        return PolicyResponse(np.asarray(result["actions"]), spec,
                              dict(result.get("metadata", {})))

    def _default_request_adapter(self, request: PolicyRequest) -> Dict[str, Any]:
        obs = request.observation
        result = {
            "observation/agentview_rgb": obs.agentview_rgb,
            "observation/wrist_rgb": obs.wrist_rgb,
            "observation/eef_pos": obs.eef_pos,
            "observation/eef_quat": obs.eef_quat,
            "observation/gripper_qpos": obs.gripper_qpos,
            "observation/joint_pos": obs.joint_pos,
            "prompt": request.instruction,
            "episode_id": request.episode_id,
            "step": request.step,
        }
        if self.action_chunk_size is not None:
            result["action_chunk_size"] = self.action_chunk_size
        return result
