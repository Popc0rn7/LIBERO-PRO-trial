"""Serial LIBERO evaluation runner."""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .action_executor import ActionChunkExecutor


@dataclass
class EpisodeResult:
    policy: str
    suite: str
    task_id: int
    task: str
    init_state_index: int
    seed: int
    success: bool
    steps: int
    elapsed_seconds: float
    inference_count: int
    failure_reason: str = ""


class EvaluationRunner:
    """Owns task iteration and failure isolation; dependencies are injectable."""

    def __init__(self, cfg, client, suite_factory=None, env_factory=None, video_writer=None):
        self.cfg = cfg
        self.client = client
        self.suite_factory = suite_factory or self._default_suite_factory
        self.env_factory = env_factory or self._default_env_factory
        self.video_writer = video_writer
        self.executor = ActionChunkExecutor(
            client,
            execute_horizon=int(cfg.rollout.execute_horizon),
            action_spec=self._action_spec(cfg.policy.action),
            clip_actions=bool(cfg.rollout.clip_actions),
        )

    @staticmethod
    def _action_spec(cfg):
        from .protocol import ActionSpec
        return ActionSpec(**dict(cfg))

    @staticmethod
    def _default_suite_factory(name):
        from libero.libero.benchmark import get_benchmark
        return get_benchmark(name)(0)

    def _default_env_factory(self, task):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        return OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=int(self.cfg.recording.camera_height),
            camera_widths=int(self.cfg.recording.camera_width),
            control_freq=int(self.cfg.policy.action.control_frequency_hz),
        )

    def run(self):
        suite_name = str(self.cfg.benchmark.suite)
        suite = self.suite_factory(suite_name)
        task_ids = list(self.cfg.benchmark.task_ids) or list(range(suite.n_tasks))
        results = []
        for task_id in task_ids:
            if task_id < 0 or task_id >= suite.n_tasks:
                raise ValueError("task_id {} outside suite range [0, {})".format(task_id, suite.n_tasks))
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            for rollout_id in range(int(self.cfg.benchmark.episodes_per_task)):
                state_index = rollout_id % len(states)
                results.append(self._run_episode(suite_name, task_id, task, state_index, states[state_index], rollout_id))
                self._append_jsonl(results[-1])
        summary = self._summary(suite_name, results)
        self._write_summary(summary)
        return summary, results

    def _run_episode(self, suite_name, task_id, task, state_index, init_state, rollout_id):
        seed = int(self.cfg.benchmark.seed) + task_id * 100000 + rollout_id
        random.seed(seed)
        np.random.seed(seed)
        env = None
        started = time.monotonic()
        steps = inference_before = 0
        success = False
        failure = ""
        try:
            env = self.env_factory(task)
            if hasattr(env, "seed"):
                env.seed(seed)
            env.reset()
            obs = env.set_init_state(init_state)
            action_cfg = self.cfg.policy.action
            action_dim = action_cfg.get("dim") if hasattr(action_cfg, "get") else action_cfg.dim
            zero = np.zeros(int(action_dim), dtype=np.float32)
            for _ in range(int(self.cfg.rollout.warmup_steps)):
                obs, _, _, _ = env.step(zero)
            episode_id = "{}/{}/{:06d}".format(suite_name, task_id, rollout_id)
            inference_before = getattr(self.client, "inference_count", 0)
            self.executor.reset(episode_id, task.language)
            deadline = started + float(self.cfg.rollout.episode_timeout_seconds)
            while steps < int(self.cfg.rollout.max_steps):
                if time.monotonic() > deadline:
                    failure = "episode_timeout"
                    break
                action = self.executor.act(obs, steps)
                obs, _, done, _ = env.step(action)
                steps += 1
                success = bool(env.check_success())
                if self.video_writer:
                    self.video_writer.append_obs(obs, success)
                if success:
                    break
                if done:
                    failure = "environment_done"
                    break
            if not success and not failure:
                failure = "max_steps"
        except Exception as exc:
            failure = "{}: {}".format(type(exc).__name__, exc)
        finally:
            if env is not None:
                env.close()
        count = getattr(self.client, "inference_count", inference_before) - inference_before
        return EpisodeResult(str(self.cfg.policy.name), suite_name, task_id, task.name, state_index, seed,
                             success, steps, time.monotonic() - started, count, failure)

    def _append_jsonl(self, result):
        path = Path(str(self.cfg.output.directory)) / str(self.cfg.output.episodes_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def _summary(self, suite, results):
        successes = sum(r.success for r in results)
        return {"policy": str(self.cfg.policy.name), "suite": suite, "episodes": len(results),
                "successes": successes, "success_rate": successes / len(results) if results else 0.0}

    def _write_summary(self, summary):
        path = Path(str(self.cfg.output.directory)) / str(self.cfg.output.summary_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
