"""Optional adapter for an OpenPI WebSocket policy server.

OpenPI is deliberately imported lazily so the LIBERO environment and model
environment remain independent.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..policy_client import PolicyClient
from ..protocol import ActionSpec, PolicyRequest, PolicyResponse
from .registry import register_client


@register_client("openpi")
class OpenPIClient(PolicyClient):
    def __init__(
        self,
        host: str,
        port: int,
        request_adapter: Optional[Callable[[PolicyRequest], Dict[str, Any]]] = None,
    ):
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise ImportError(
                "OpenPIClient requires openpi-client in the LIBERO environment"
            ) from exc
        self._policy = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._request_adapter = request_adapter or self._default_request_adapter

    @classmethod
    def from_config(cls, cfg):
        connection = cfg.get("connection", {})
        host = connection.get("host")
        port = connection.get("port")
        if not host or port is None:
            raise ValueError("openpi connection.host and connection.port are required")
        return cls(str(host), int(port))

    def close(self):
        close = getattr(self._policy, "close", None)
        if close:
            close()

    def reset(self, episode_id: str, instruction: str) -> None:
        # The public OpenPI client has no universal reset RPC. Episode identity
        # is sent on every inference; deploy a server-side reset hook if the
        # selected policy keeps temporal state.
        self._episode_id = episode_id

    def infer(self, request: PolicyRequest) -> PolicyResponse:
        result = self._policy.infer(self._request_adapter(request))
        if "actions" not in result:
            raise ValueError("OpenPI response does not contain 'actions'")
        spec_dict = result.get("action_spec", {})
        action_spec = ActionSpec(**spec_dict) if spec_dict else request.action_spec
        return PolicyResponse(
            actions=np.asarray(result["actions"], dtype=np.float32),
            action_spec=action_spec,
            metadata=dict(result.get("metadata", {})),
        )

    @staticmethod
    def _default_request_adapter(request: PolicyRequest) -> Dict[str, Any]:
        obs = request.observation
        return {
            # These canonical keys require a matching transform in the OpenPI
            # server config. Pass request_adapter for a checkpoint-specific schema.
            "observation/agentview_rgb": obs.agentview_rgb,
            "observation/wrist_rgb": obs.wrist_rgb,
            "observation/eef_pos": obs.eef_pos,
            "observation/eef_quat": obs.eef_quat,
            "observation/gripper_qpos": obs.gripper_qpos,
            "observation/joint_pos": obs.joint_pos,
            "prompt": request.instruction,
            "episode_id": request.episode_id,
            "step": request.step,
            "protocol_version": request.protocol_version,
        }
