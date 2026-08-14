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
  rollout.execute_horizon=8 \
  benchmark.evaluation_config_path=evaluation_config.yaml benchmark.task_ids='[0,1]' \
  benchmark.init_state_ids='[0,2]' \
  recording.enabled=true live_preview.enabled=true \
  output.directory=outputs/my_eval
```

默认使用 MuJoCo EGL 离屏渲染。只有系统安装了 OSMesa 时才覆盖 `environment.render_backend=osmesa`。

`benchmark.init_state_ids=[]` 时，每个 task 根据 `episodes_per_task` 顺序选择不重复初态；默认配置只运行 1 个初态，方便 smoke test。完整评测可设置 `benchmark.episodes_per_task=50`。请求数量超过可用初态时会报错，不会循环复用。显式提供列表时，该列表直接决定 episode schedule 和数量。每个 task 环境固定使用 `benchmark.seed`（默认 7），不会在 episode 之间重新派生 seed。

## Perturbation

扰动由外部 `evaluation_config.yaml` 统一配置，Evaluator 只接收它的路径：

```bash
python -m libero.evaluation.eval \
  benchmark.evaluation_config_path=evaluation_config.yaml
```

配置中用 `task_suite_name` 指定基础 suite，并通过 `use_*` 开启扰动：

```yaml
task_suite_name: libero_goal

use_environment: true
use_swap: false
use_object: false
use_language: false
use_task: false

perturbation_mapping:
  use_environment: env
  use_swap: swap
  use_object: object
  use_language: lan
  use_task: task
```

单扰动的最终 suite 名称由 `task_suite_name` 和 `perturbation_mapping` 组成，例如上面的配置对应 `libero_goal_env`。多个非 task 扰动组合时使用 `libero_goal_temp`；`use_task` 不能与其他扰动同时启用。

启动时，[`perturbation.py`](perturbation.py) 会读取这份配置：

- BDDL 和 `.pruned_init` 都存在时直接复用；
- 只有 `.pruned_init` 缺失时，仅运行 `generate_init_states.py`；
- BDDL 缺失时，先运行对应 Perturbator，再生成 `.pruned_init`。

生成完成后，任务、语言、目标条件和初始状态均由 LIBERO 原生 benchmark 和 BDDL parser 加载。根目录的 `perturbation.py` 仅保留为原仓库示例，evaluation 不会调用它。

## Helper

### Mock Policy Server

可以使用 mock 的 policy 来离线测试评测联通性，默认 `noop` mock 保持机械臂不动，夹爪重复开合。

```bash
python -m libero.evaluation.mock_server --mode noop --chunk-size 16
```

### Record & Live-Preview & Video Render

通过开启 `configs/eval.yaml` 中 `recording` 和 `live_preview` 可以打开视频保存和浏览器在线预览功能。

为提高录制视频的可视化效果，额外提供一个脚本加工录制的视频，Usage: 

```bash
python scripts/render_eval_videos.py \
    outputs/2026-08-14/11-16-46/evaluation
```

## 架构

Evaluator 先准备扰动数据，再通过 LIBERO 创建环境并执行 rollout。`PolicyClient` 负责与策略服务通信及模型输入输出转换。

```text
evaluation_config.yaml
        ↓
perturbation.py（复用或生成 BDDL / .pruned_init）
        ↓
LIBERO benchmark + environment
        ↓ observation                         ↑ action
EvaluationRunner → ActionChunkExecutor → PolicyClient → Policy Server
```

Evaluator 管理环境生命周期、episode schedule、action chunk 执行、录像和结果统计。`PolicyClient` 管理连接、wire protocol、图像与状态预处理及 action decoding；双方只通过 `PolicyRequest` 和 `PolicyResponse` 交换数据。

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

Libero-pro协议定义在 `protocol.py` ，规范 evaluator 内部数据结构。Client 完全控制 wire protocol、传输、序列化、图像与 state 预处理、归一化和 action decoding。Evaluator 通过 `rollout.execute_horizon` 决定一个 chunk 最多执行多少个动作；OpenPI 返回的 chunk 长度由服务端模型配置决定。

Action 必须是非空、有限、位于 `[-1, 1]` 的 `float32[T, 7]`，类型为 `delta_ee`。非法响应只终止当前 episode；OpenPI 的连接和请求行为由官方 `WebsocketClientPolicy` 管理，episode 总 timeout 来自 `rollout.episode_timeout_seconds`。

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
