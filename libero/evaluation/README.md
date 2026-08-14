# LIBERO Policy Evaluation

## Quick Start

安装额外需要的依赖：

```bash
conda activate libero_pro
pip install -r libero/evaluation/requirements.txt
```

运行测评：

```bash
python -m libero.evaluation.eval policy=mock benchmark.task_ids='[0]' benchmark.episodes_per_task=1
```

## 常用配置

```bash
python -m libero.evaluation.eval policy=pi0 \
  policy.connection.host=127.0.0.1 policy.connection.port=8000 \
  policy.inference.action_chunk_size=16 rollout.execute_horizon=8 \
  benchmark.suite=libero_goal benchmark.task_ids='[0,1]' \
  benchmark.init_state_ids='[0,2]' \
  recording.enabled=true live_preview.enabled=true \
  output.directory=outputs/my_eval
```

默认使用 MuJoCo EGL 离屏渲染。只有系统安装了 OSMesa 时才覆盖 `environment.render_backend=osmesa`。

`benchmark.init_state_ids=[]` 时，每个 task 根据 `episodes_per_task` 顺序选择初态；显式提供列表时，该列表直接决定 episode schedule 和数量。

## Helper

### Mock Policy Server

可以使用 mock 的 policy 来离线测试评测联通性，默认 `noop` mock 保持机械臂不动，夹爪重复开合。

```bash
python -m libero.evaluation.mock_server --mode noop --chunk-size 16
```

### Record & Live-Preview & Video Render

通过开启 `config/eval.yaml` 中 `recording` 和 `live_preview` 可以打开视频保存和浏览器在线预览功能。

为提高录制视频的可视化效果，额外提供一个脚本加工录制的视频，Usage: 

```bash
python scripts/render_eval_videos.py \
    outputs/2026-08-14/11-16-46/evaluation
```

## 架构

Evaluator 负责 Hydra 配置、LIBERO task/init state、环境生命周期、同步 rollout、action 执行上限、录像、预览和结果统计。`PolicyClient` 负责连接策略服务、健康检查、超时、wire protocol、序列化、模型预处理和 action decoding。两者只通过 `PolicyRequest`、`PolicyResponse`、`RawObservation` 和 `ActionSpec` 交换内部 Python 数据。

```text
Hydra CLI → EvaluationRunner → ActionChunkExecutor → PolicyClient → Policy Server
                    ↓                                      ↑
             LIBERO environment              WebSocket / Client-owned protocol
```

Evaluator 不规定 HTTP/WebSocket 字段，不翻转或缩放输入图像，不修改四元数或拼接 state vector，也不裁剪 action。Client 返回的 action 必须已经能够直接传给 `env.step()`。

## 新增 Policy

### A. 复用已有的传输协议

比如使用 OpenPI 系列的 policy，只需复制 `configs/policy/pi0.yaml` 为 `configs/policy/<name>.yaml`，修改连接和推理配置，随后以 `policy=<name>` 运行。

### B. 使用新的传输协议

新增传输协议应新增client负责数据翻译 `clients/<name>_client.py`，用 `@register_client("<name>")` 注册并实现：

```python
class MyClient(PolicyClient):
    @classmethod
    def from_config(cls, cfg): ...
    def check(self) -> ClientInfo: ...
    def reset(self, episode_id, instruction): ...
    def infer(self, request) -> PolicyResponse: ...
    def close(self): ...
```

Libero-pro协议定义在 `protocol.py` ，规范 evaluator 内部数据结构。Client 完全控制 wire protocol、传输、序列化、图像与 state 预处理、归一化和 action decoding。`policy.inference.action_chunk_size` 只提供给 Client；Evaluator 仅通过 `rollout.execute_horizon` 决定一个 chunk 最多执行多少个动作。

Action 必须是非空、有限、位于 `[-1, 1]` 的 `float32[T, 7]`，类型为 `delta_ee`。非法响应只终止当前 episode；单次请求 timeout 来自 `policy.connection.timeout_seconds`，episode 总 timeout 来自 `rollout.episode_timeout_seconds`。

## 常见问题

### PyTorch 无法加载 executable stack

如果出现 `libtorch_cpu.so: cannot enable executable stack`，可使用仓库脚本清除错误 ELF 标志；脚本会先创建备份：

```bash
python scripts/clear_elf_execstack.py
```

### OSMesa 无法加载

如果出现 `libOSMesa.so.0: cannot open shared object file`，使用本机已验证的 EGL：

```bash
python -m libero.evaluation.eval policy=mock environment.render_backend=egl
```
