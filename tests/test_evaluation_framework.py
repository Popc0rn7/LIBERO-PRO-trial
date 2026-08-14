import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from libero.evaluation import (ActionSpec, ClientInfo, EvaluationRunner, PolicyClient,
                               PolicyRequest, PolicyResponse, RawObservation)
from libero.evaluation.clients import available_clients, create_client, register_client
from libero.evaluation.clients.openpi_client import OpenPIClient
from libero.evaluation.policy_client import PolicyClient


def test_builtin_registry_and_unknown_client():
    assert "openpi" in available_clients()
    with pytest.raises(ValueError, match="unknown client"):
        create_client({"client": "missing"})


def test_duplicate_registration_is_rejected():
    name = "test_duplicate_client"

    @register_client(name)
    class Client(PolicyClient):
        def reset(self, episode_id, instruction): pass
        def infer(self, request): pass

    with pytest.raises(ValueError, match="already registered"):
        register_client(name)(Client)


@pytest.mark.parametrize("actions,message", [
    (np.zeros(7, np.float32), "shape"),
    (np.zeros((0, 7), np.float32), "non-empty"),
    (np.zeros((2, 6), np.float32), "action_dim"),
    (np.full((2, 7), np.nan, np.float32), "NaN"),
])
def test_action_contract_rejections(actions, message):
    with pytest.raises(ValueError, match=message):
        PolicyResponse(actions).validate(ActionSpec())


def test_action_contract_preserves_openpi_float64_actions():
    actions = np.full((2, 7), 1.5, dtype=np.float64)
    response = PolicyResponse(actions).validate(ActionSpec())
    assert response.actions.dtype == np.float64
    np.testing.assert_array_equal(response.actions, actions)


def test_openpi_libero_request_matches_upstream_protocol():
    # Non-symmetric pixels prove that preprocessing rotates by 180 degrees.
    image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    wrist = image + 20
    obs = RawObservation(
        agentview_rgb=image,
        wrist_rgb=wrist,
        eef_pos=np.array([1, 2, 3], dtype=np.float32),
        # xyzw quaternion for a 90-degree rotation around z.
        eef_quat=np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32),
        gripper_qpos=np.array([0.1, 0.2], dtype=np.float32),
        joint_pos=np.arange(7, dtype=np.float32),
    )
    request = PolicyRequest("episode", 4, "open drawer", obs)
    payload = OpenPIClient("localhost", 8000, image_size=2)._default_request_adapter(request)

    assert set(payload) == {
        "observation/image", "observation/wrist_image", "observation/state", "prompt"
    }
    assert payload["observation/image"].shape == (2, 2, 3)
    assert payload["observation/image"].dtype == np.uint8
    # A square input stays unpadded, exposing the reference 180-degree rotation.
    np.testing.assert_array_equal(payload["observation/image"][0, 0], image[-1, -1])
    assert payload["observation/state"].shape == (8,)
    np.testing.assert_allclose(payload["observation/state"][:3], [1, 2, 3])
    np.testing.assert_allclose(payload["observation/state"][3:6], [0, 0, np.pi / 2], rtol=1e-6)
    np.testing.assert_allclose(payload["observation/state"][6:], [0.1, 0.2])
    assert payload["prompt"] == "open drawer"


def test_openpi_response_maps_upstream_server_timing():
    result = {
        "actions": np.zeros((16, 7)),
        "server_timing": {"infer_ms": 91.25, "prev_total_ms": 95.0},
        "policy_timing": {"infer_ms": 88.5},
    }
    metadata = OpenPIClient._response_metadata(result)
    assert metadata["server_inference_latency_ms"] == 91.25
    assert metadata["server_timing"]["prev_total_ms"] == 95.0
    assert metadata["policy_timing"]["infer_ms"] == 88.5


class Task:
    name = "task"
    language = "do task"


class Suite:
    n_tasks = 2
    def get_task(self, task_id): return Task()
    def get_task_init_states(self, task_id): return [np.zeros(1)]


class Env:
    closed = 0
    action_types = []
    def seed(self, seed): pass
    def reset(self): return None
    def set_init_state(self, state): return _obs()
    def step(self, action):
        Env.action_types.append(type(action))
        return _obs(), 0, False, {}
    def check_success(self): return True
    def close(self): Env.closed += 1


def _obs():
    return {"agentview_image": np.zeros((2, 2, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
            "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.zeros(2), "robot0_joint_pos": np.zeros(7)}


def test_runner_multiple_tasks_outputs_and_closes(tmp_path):
    Env.action_types = []
    policy = NS(name="mock", action={"type": "delta_ee", "dim": 7,
                                      "controller": "OSC_POSE", "control_frequency_hz": 20})
    cfg = NS(policy=policy, benchmark=NS(suite="suite", task_ids=[0, 1], episodes_per_task=1, init_state_ids=[], seed=3),
             rollout=NS(execute_horizon=2, warmup_steps=0, max_steps=3, episode_timeout_seconds=10),
             recording=NS(enabled=False, directory=str(tmp_path/"videos"), fps=10, stride=1),
             live_preview=NS(stride=1),
             output=NS(directory=str(tmp_path), episodes_file="episodes.jsonl", summary_file="summary.json"))
    class Client(PolicyClient):
        def check(self): return ClientInfo(True, "test")
        def reset(self, episode_id, instruction): pass
        def infer(self, request): return PolicyResponse(np.zeros((2, 7), np.float32))
    client = Client()
    summary, results = EvaluationRunner(cfg, client, lambda _: Suite(), lambda _: Env()).run()
    assert summary["success_rate"] == 1.0 and len(results) == 2
    assert Env.closed == 2
    assert Env.action_types and all(action_type is list for action_type in Env.action_types)
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "summary.json").read_text())["successes"] == 2
    episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert episode["prompt"] == "do task"
    assert episode["video_path"] == ""
    assert episode["wrist_video_path"] == ""
