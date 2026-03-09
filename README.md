# ULGP Plugin

ULGP (Uncertainty-guided Local Graph Propagation) 是一个 test-time 异常图后处理插件：

- 无需重训练
- 不修改原模型主体
- 可直接挂在任意 anomaly map 模型后面

## 可行性评价

你的方案是可行的，且适合跨模型插件化。当前实现重点做了工程收敛：

- 策略可插拔：uncertainty / graph / fusion 都是策略接口。
- 明确对齐层：新增 `preprocess.py`，统一 dtype/device/shape 与 graph space 对齐。
- 三模式统一：`full / feature-lite / map-only`。
- 可诊断：`refine(..., return_info=True)` 返回关键诊断字段。
- 可运行：包含 CLI、configs、examples、tests。

## 目录结构

```text
ULGP/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    default.yaml
    musc.yaml
    anomalyclip.yaml
    patchcore.yaml
  examples/
    demo_map_only.py
    demo_feature_lite.py
    demo_full.py
    integrate_musc.py
    integrate_anomalyclip.py
  tests/
    test_candidate.py
    test_uncertainty.py
    test_graph.py
    test_propagator.py
    test_plugin_modes.py
  ulgp/
    __init__.py
    utils.py
    preprocess.py
    candidate.py
    uncertainty.py
    graph.py
    propagator.py
    fusion.py
    plugin.py
    vis.py
    cli.py
```

## 关键模块职责

- `preprocess.py`
  - `prepare_plugin_inputs(...)`: 统一输入格式，自动选择 graph space（feature 或 map）
  - `restore_output_format(...)`: 将 graph space 输出映射回原始 A0 空间
- `uncertainty.py`
  - `PrecomputedUncertainty`
  - `FeatureInstabilityUncertainty`
  - `ScoreVarianceUncertainty`
- `graph.py`
  - `FeatureKNNGraphBuilder`
  - `GridGraphBuilder`
- `fusion.py`
  - `LinearFusion`
  - `IdentityFusion`
- `plugin.py`
  - 仅负责 orchestration（mode 解析、策略选择、流程调度）

## 输入输出约定

- `A0`（必需）：原模型 anomaly map，`(B,1,H,W)` 或 `(B,H,W)`
- `F`（可选）：特征图，`(B,C,h,w)` 或 `(B,N,C)`
- `U`（可选）：不确定性图，`(B,1,H,W)` 或 `(B,H,W)`
- `I`（可选）：原图，`(B,3,H,W)`
- 输出：`A_ref`，与 `A0` 同 shape

## 统一接口

```python
from ulgp import ULGPPlugin

plugin = ULGPPlugin(mode="auto")
A_ref, info = plugin.refine(
    anomaly_map=A0,
    feature_map=F,
    uncertainty_map=U,
    image=I,
    return_info=True,
)
```

`info` 包含：

- `mode`
- `graph_space`
- `candidate_ratio_actual`
- `uncertainty_stats`
- `num_nodes`, `num_edges`
- `mean_update`, `max_update`
- `propagation_steps`
- `runtime_ms`

## CLI

```bash
python -m ulgp.cli \
  --a0 data/a0.npy \
  --feat data/feat.npy \
  --u data/u.npy \
  --img data/img.npy \
  --out data/a_ref.npy \
  --config configs/default.yaml \
  --save_info \
  --save_u \
  --save_candidate \
  --save_vis
```

说明：

- `--config` 支持 YAML（安装 `pyyaml`）或 JSON
- `--save_info` 导出诊断信息 json
- `--save_u` 导出 uncertainty map
- `--save_candidate` 导出 candidate mask
- `--save_vis` 导出 `a0/ref` 可视化图

## 运行示例

```bash
python examples/demo_map_only.py
python examples/demo_feature_lite.py
python examples/demo_full.py
```

## 测试

```bash
pytest -q tests
```
