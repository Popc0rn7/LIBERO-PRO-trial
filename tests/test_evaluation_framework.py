import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from libero.evaluation import ActionSpec, ClientInfo, EvaluationRunner, PolicyClient, PolicyResponse
from libero.evaluation.clients import available_clients, create_client, register_client
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


class Task:
    name = "task"
    language = "do task"


class Suite:
    n_tasks = 2
    def get_task(self, task_id): return Task()
    def get_task_init_states(self, task_id): return [np.zeros(1)]


class Env:
    closed = 0
    def seed(self, seed): pass
    def reset(self): return None
    def set_init_state(self, state): return _obs()
    def step(self, action): return _obs(), 0, False, {}
    def check_success(self): return True
    def close(self): Env.closed += 1


def _obs():
    return {"agentview_image": np.zeros((2, 2, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
            "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.zeros(2), "robot0_joint_pos": np.zeros(7)}


def test_runner_multiple_tasks_outputs_and_closes(tmp_path):
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
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "summary.json").read_text())["successes"] == 2
