# LIBERO Policy Evaluation

## CLI usage

```bash
conda activate libero_pro
python -m libero.evaluation.eval \
  policy=pi0 \
  benchmark.suite=libero_goal \
  benchmark.task_ids='[0,1]' \
  benchmark.episodes_per_task=10
```

Common overrides:

```bash
# Policy server address
policy.connection.host=127.0.0.1
policy.connection.port=8000

# Evaluation settings
benchmark.seed=0
rollout.execute_horizon=8
rollout.warmup_steps=10
rollout.max_steps=600
rollout.episode_timeout_seconds=300

# Output
output.directory=outputs/my_eval
recording.save_video=true
```

Use the mock policy to check the CLI and evaluation pipeline without starting a
model server:

```bash
python -m libero.evaluation.eval \
  policy=mock \
  benchmark.task_ids='[0]'
```

Results are written to `<output.directory>/episodes.jsonl` and
`<output.directory>/summary.json`.

The complete list of evaluation options and their defaults is in
[`configs/eval.yaml`](configs/eval.yaml). Policy-specific options are in
[`configs/policy/`](configs/policy/).

## Add a policy

First decide whether the policy can use an existing client.

### Option A: reuse an existing client

If the model is served by OpenPI, only add one YAML file. Copy
[`configs/policy/pi0.yaml`](configs/policy/pi0.yaml) to
`configs/policy/<policy_name>.yaml` and update its values:

```yaml
name: my_policy
client: openpi

connection:
  host: 127.0.0.1
  port: 8000
  timeout_seconds: 30

protocol:
  version: libero-policy-v1

action:
  type: delta_ee
  dim: 7
  controller: OSC_POSE
  control_frequency_hz: 20

capabilities:
  action_chunk: true
  stateful: false
```

Run it using the YAML filename:

```bash
python -m libero.evaluation.eval policy=my_policy
```

Reference files:

- OpenPI policy config: [`configs/policy/pi0.yaml`](configs/policy/pi0.yaml)
- OpenPI client implementation: [`clients/openpi_client.py`](clients/openpi_client.py)


### Option B: add a new client

If the policy uses another server protocol, make these three changes:

1. Add `clients/<client_name>_client.py`.
2. Import the new class in [`clients/__init__.py`](clients/__init__.py).
3. Add `configs/policy/<policy_name>.yaml` with `client: <client_name>`.

Use [`clients/openpi_client.py`](clients/openpi_client.py) as a network-client
example. A client must register itself and implement four methods:

```python
from libero.evaluation import PolicyClient, PolicyResponse, register_client


@register_client("my_client")
class MyClient(PolicyClient):
    @classmethod
    def from_config(cls, cfg):
        # Read cfg.connection and create the transport.
        return cls(...)

    def reset(self, episode_id, instruction):
        # Clear state from the previous episode.
        ...

    def infer(self, request):
        # Send request.observation and request.instruction to the policy.
        return PolicyResponse(actions=...)

    def close(self):
        # Close sockets or other resources.
        ...
```

Before implementing `infer()`, follow the protocol below. The Python source of
truth is [`protocol.py`](protocol.py).

## Policy protocol: `libero-policy-v1`

This section is the contract between the evaluator and every policy client.
Transport adapters may choose their own wire encoding, but they must preserve
these fields and semantics when constructing `PolicyRequest` and
`PolicyResponse`.

### Episode lifecycle

For every episode, the evaluator performs the following calls in order:

1. `client.reset(episode_id, instruction)` exactly once.
2. `client.infer(request)` whenever no cached action remains.
3. `client.close()` after the complete evaluation finishes.

`episode_id` is unique and has the form `<suite>/<task_id>/<rollout_id>`.
A stateful policy must discard all state from the previous episode during
`reset()`. The current OpenPI transport does not provide a reset RPC; therefore
it should only be used with stateless servers unless the deployment adds its
own reset handling.

### Inference request

`infer()` receives a [`PolicyRequest`](protocol.py) with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `protocol_version` | `str` | Always `libero-policy-v1`. |
| `episode_id` | `str` | Stable unique ID for the current episode. |
| `step` | `int` | Environment step at which this chunk is requested. |
| `instruction` | `str` | LIBERO natural-language task instruction. |
| `observation` | `RawObservation` | Latest unprocessed LIBERO observation. |
| `action_spec` | `ActionSpec` | Action format required by the evaluator. |
| `metadata` | `dict` | Reserved extension field; currently empty. |

`RawObservation` contains:

| Field | dtype | Shape / convention |
| --- | --- | --- |
| `agentview_rgb` | `uint8` | `[H, W, 3]`, raw `agentview_image`. |
| `wrist_rgb` | `uint8` | `[H, W, 3]`, raw `robot0_eye_in_hand_image`. |
| `eef_pos` | `float32` | `[3]`, end-effector position. |
| `eef_quat` | `float32` | `[4]`, LIBERO/robosuite quaternion as provided by the environment. |
| `gripper_qpos` | `float32` | `[2]`, raw gripper joint positions. |
| `joint_pos` | `float32` | `[7]`, raw robot joint positions. |

The evaluator intentionally does not flip or resize images, change quaternion
representations, concatenate state, or normalize values. Those operations are
checkpoint-specific and belong in the policy service.

The currently supported `ActionSpec` is:

```text
type: delta_ee
dim: 7
controller: OSC_POSE
control_frequency_hz: 20
```

### Inference response

`infer()` must return a [`PolicyResponse`](protocol.py):

| Field | Type | Requirement |
| --- | --- | --- |
| `actions` | array-like | Non-empty `float32[T, 7]` action chunk. |
| `action_spec` | `ActionSpec` | Must use the requested action type. |
| `metadata` | `dict` | Optional diagnostics; ignored by rollout logic. |

Every row of `actions` must already be decoded, de-normalized, and directly
executable by `env.step()`. The evaluator rejects an empty chunk, a one-
dimensional action, a dimension other than 7, a mismatched action type, and any
NaN or infinite value. If `rollout.clip_actions=true`, accepted actions are
clipped elementwise to `[-1, 1]` immediately before execution.

The evaluator executes at most `rollout.execute_horizon` rows from a response.
If the response is shorter, it requests another chunk after consuming all rows;
if it is longer, rows beyond the horizon are discarded. A new request always
contains the latest observation.

### OpenPI wire mapping

[`clients/openpi_client.py`](clients/openpi_client.py) maps the protocol request
to the following OpenPI dictionary keys:

```text
observation/agentview_rgb
observation/wrist_rgb
observation/eef_pos
observation/eef_quat
observation/gripper_qpos
observation/joint_pos
prompt
episode_id
step
protocol_version
```

The OpenPI server response must contain `actions`. It may also contain
`action_spec` and `metadata`; when `action_spec` is omitted, the client uses the
specification from the request.
