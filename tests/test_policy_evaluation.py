import unittest

import numpy as np

from libero.evaluation import ActionChunkExecutor, PolicyClient, PolicyResponse


def observation():
    return {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_joint_pos": np.zeros(7),
    }


class FakeClient(PolicyClient):
    def __init__(self):
        self.calls = 0

    def reset(self, episode_id, instruction):
        pass

    def infer(self, request):
        self.calls += 1
        return PolicyResponse(np.full((4, 7), self.calls, dtype=np.float32))


class ActionChunkExecutorTest(unittest.TestCase):
    def test_replans_at_shared_horizon_and_clips(self):
        client = FakeClient()
        executor = ActionChunkExecutor(client, execute_horizon=2)
        executor.reset("episode-1", "do it")
        values = [executor.act(observation(), step)[0] for step in range(3)]
        self.assertEqual(values, [1.0, 1.0, 1.0])
        self.assertEqual(client.calls, 2)

    def test_reset_discards_old_chunk(self):
        client = FakeClient()
        executor = ActionChunkExecutor(client, execute_horizon=3, clip_actions=False)
        executor.reset("episode-1", "first")
        executor.act(observation(), 0)
        executor.reset("episode-2", "second")
        self.assertEqual(executor.act(observation(), 0)[0], 2.0)


if __name__ == "__main__":
    unittest.main()
