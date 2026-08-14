"""Serial LIBERO evaluation runner with task-level environment reuse."""

import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .action_executor import ActionChunkExecutor
from .record import VideoRecorder, upright_rgb


def latency_stats(values):
    values = list(map(float, values))
    return {"count": len(values), "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "max_ms": max(values) if values else None}


@dataclass
class EpisodeResult:
    policy: str; suite: str; task_id: int; task: str; episode_id: str
    init_state_id: int; seed: int; success: bool; steps: int; duration_seconds: float
    policy_query_count: int; termination_reason: str = ""
    round_trip_latency_ms: list = field(default_factory=list)
    server_inference_latency_ms: list = field(default_factory=list)
    video_path: str = ""


class EvaluationRunner:
    def __init__(self, cfg, client, suite_factory=None, env_factory=None, preview=None):
        self.cfg, self.client, self.preview = cfg, client, preview
        self.suite_factory = suite_factory or self._default_suite_factory
        self.env_factory = env_factory or self._default_env_factory
        self.executor = ActionChunkExecutor(client, int(cfg.rollout.execute_horizon), self._action_spec(cfg.policy.action))

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
        return OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=int(self.cfg.recording.camera_height), camera_widths=int(self.cfg.recording.camera_width),
            horizon=int(self.cfg.rollout.max_steps) + int(self.cfg.rollout.warmup_steps) + 20, ignore_done=True)

    def run(self):
        suite_name = str(self.cfg.benchmark.suite); suite = self.suite_factory(suite_name)
        task_ids = list(self.cfg.benchmark.task_ids) or list(range(suite.n_tasks)); results = []
        for task_id in task_ids:
            if not 0 <= task_id < suite.n_tasks: raise ValueError("task_id out of range: {}".format(task_id))
            task, states = suite.get_task(task_id), suite.get_task_init_states(task_id)
            schedule = list(self.cfg.benchmark.init_state_ids)
            if not schedule:
                if not len(states): raise ValueError("task has no init states")
                schedule = [i % len(states) for i in range(int(self.cfg.benchmark.episodes_per_task))]
            if any(i < 0 or i >= len(states) for i in schedule): raise ValueError("init_state_id out of range")
            env = None
            try:
                env = self.env_factory(task)
                for episode_index, state_id in enumerate(schedule):
                    result = self._run_episode(env, suite_name, task_id, task, episode_index, state_id, states[state_id])
                    results.append(result); self._append_jsonl(result); self._write_summary(self._summary(suite_name, results))
            finally:
                if env is not None: env.close()
        summary = self._summary(suite_name, results); self._write_summary(summary); return summary, results

    def _publish(self, obs, status):
        if self.preview: self.preview.publish(agentview_rgb=upright_rgb(obs, "agentview_image"), wrist_rgb=upright_rgb(obs, "robot0_eye_in_hand_image"), status=status)

    def _run_episode(self, env, suite, task_id, task, episode_index, state_id, state):
        seed = int(self.cfg.benchmark.seed) + task_id * 100000 + episode_index
        random.seed(seed); np.random.seed(seed); started = time.monotonic(); steps = 0; success = False; reason = ""
        episode_id = "{}/{}/{}".format(suite, task_id, episode_index)
        video_path = Path(str(self.cfg.recording.directory)) / "task_{}_episode_{}_init_{}.mp4".format(task_id, episode_index, state_id)
        try:
            if hasattr(env, "seed"): env.seed(seed)
            env.reset(); obs = env.set_init_state(state)
            warmup = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
            for warmup_step in range(int(self.cfg.rollout.warmup_steps)):
                obs, _, _, _ = env.step(warmup)
                if warmup_step % int(self.cfg.live_preview.stride) == 0: self._publish(obs, {"phase":"warmup", "episode_id":episode_id})
            self.executor.reset(episode_id, task.language)
            with VideoRecorder(video_path, bool(self.cfg.recording.enabled), int(self.cfg.recording.fps)) as video:
                deadline = time.monotonic() + float(self.cfg.rollout.episode_timeout_seconds)
                last_recorded = False
                while steps < int(self.cfg.rollout.max_steps):
                    if time.monotonic() >= deadline: reason = "episode_timeout"; break
                    action = self.executor.act(obs, steps); obs, _, _, _ = env.step(action.tolist()); steps += 1
                    success = bool(env.check_success())
                    if (steps - 1) % int(self.cfg.recording.stride) == 0: video.append(upright_rgb(obs, "agentview_image")); last_recorded = True
                    else: last_recorded = False
                    if (steps - 1) % int(self.cfg.live_preview.stride) == 0: self._publish(obs, {"phase":"rollout", "step":steps, "success":success})
                    if success: reason = "success"; break
                if not success and not reason: reason = "max_steps"
                if steps and not last_recorded: video.append(upright_rgb(obs, "agentview_image"))
                self._publish(obs, {"phase":"episode_complete", "step":steps, "success":success, "termination_reason":reason})
        except Exception as exc:
            reason = "{}: {}".format(type(exc).__name__, exc)
        return EpisodeResult(str(self.cfg.policy.name), suite, task_id, task.name, episode_id, state_id, seed,
            success, steps, time.monotonic()-started, self.executor.query_count, reason,
            list(self.executor.round_trip_latency_ms), list(self.executor.server_inference_latency_ms),
            str(video_path) if bool(self.cfg.recording.enabled) else "")

    def _append_jsonl(self, result):
        path = Path(str(self.cfg.output.directory)) / str(self.cfg.output.episodes_file); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    def _summary(self, suite, results):
        success = sum(r.success for r in results); rt = sum((r.round_trip_latency_ms for r in results), []); server = sum((r.server_inference_latency_ms for r in results), [])
        return {"policy":str(self.cfg.policy.name), "suite":suite, "episodes":len(results), "successes":success,
            "success_rate":success/len(results) if results else 0.0, "policy_query_count":sum(r.policy_query_count for r in results),
            "round_trip_latency":latency_stats(rt), "server_inference_latency":latency_stats(server)}
    def _write_summary(self, summary):
        path = Path(str(self.cfg.output.directory)) / str(self.cfg.output.summary_file); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8"); os.replace(str(tmp), str(path))
