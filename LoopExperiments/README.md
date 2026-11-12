# LoopExperiments：Ouro 能否在第 2 轮确定第 4 轮的输出？

- 研究计划：[plan.md](plan.md)
- 可执行的实验设计与已完成的验证：[EXPERIMENT.md](EXPERIMENT.md)

**原始项目状态**：代码、单元测试、前向等价性验证与三数据集冒烟测试均已完成；原计划的 2,197 题全量采集尚未启动。

后续已完成的受控实验：

- [依赖深度与分支状态替换](dag_depth/RESULTS.md)：Ouro-2.6B 上整题的并行分支合并困难。
- [两路状态提前汇合](early_merge/RESULTS.md)：第 1 轮 latent 状态可利用，但没有超过线性读出加文本合并。
- [按分支动态分配 loop](agentic_loop_depth/RESULTS.md)：原提示下平均延迟优于固定 3 轮；换 few-shot 后发现高置信错误。它仍是给定分工的模型微基准，不是完整 main/subagent 工作流。

## 布局

| 路径 | 内容 |
|---|---|
| `src/loopexp/bounded.py` | 截断、$m_2$、紧凑统计与充分性证书、tie-breaking |
| `src/loopexp/ouro.py` | 重写的 Ouro 批量解码：每轮独立 KV，记录第 2、4 轮的 FP32 读出 |
| `src/loopexp/data.py` | 数据加载、按题划分校准/测试、chat 提示、准确率核对 |
| `scripts/collect.py` | 采集，可断点续跑，OOM 时自动拆小 batch |
| `scripts/run_full.sh` | 多卡全量采集 |
| `scripts/analyze.py` | 校准、四种方法、bootstrap CI、曲线、图、自检、`report.md` |
| `scripts/validate_forward.py` | 与官方前向对比、批量与单条对比 |
| `tests/` | 数学性质与合成数据端到端测试 |
| `results/validate_forward.json` | 前向等价性验证结果 |
| `runs/smoke/` | 冒烟数据与分析（每数据集 16 道校准题，仅用于检查流程） |

## 环境

`.venv` 是指向 `~/.venvs/loopexp` 的符号链接（放在本地盘：在 NFS 上安装 torch 会超时）。主要版本：Python 3.12、torch 2.8.0+cu128、transformers 4.55.0（Ouro 远程代码要求低于 4.56）。重建方法：

```bash
uv venv --python 3.12 ~/.venvs/loopexp && ln -sfn ~/.venvs/loopexp .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python transformers==4.55.0 numpy pandas pyarrow matplotlib safetensors huggingface_hub math-verify pytest
```

模型权重：`/mnt/disk2_from_server2/siqizhu4/models/ouro-1.4b`（rev `574fa66`）。原始数据：`data/raw/`。
