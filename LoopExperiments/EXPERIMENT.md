# 第一阶段实验设计：Ouro-1.4B 第 2 轮可证明退出 + 有界修正

本文把 [plan.md](plan.md) 落实为可执行协议。plan.md 未定的细节在这里给出具体选择及理由。代码与本文件不一致时，以代码为准，并回来修正本文件。

## 0. 一句话

用原始四轮 Ouro-1.4B 在 GSM8K、MATH-500 和 MBPP+ 上做 greedy 解码。在每个生成位置记录第 2、4 轮的 FP32 logits 统计。之后离线计算任意 $B$ 下四种方法的输出，与原始四轮输出比较。第一阶段只用一次生成轨迹，不训练，也不做自由生成。

## 1. 模型与读出

| 项 | 选择 | 理由 |
|---|---|---|
| 模型 | `ByteDance/Ouro-1.4B`，rev `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`（与 loop_suffix 项目同版本） | plan 指定；固定版本便于复现 |
| 本地路径 | `/mnt/disk2_from_server2/siqizhu4/models/ouro-1.4b` | |
| 轮数 | 固定 4 轮，**不经过 early-exit gate** | 官方 `OuroForCausalLM` 默认 `early_exit_threshold=1.0`，bf16 下 sigmoid 饱和可能提前退出。本实验直接读第 4 轮，保证“原始四轮” |
| 第 $d$ 轮读出 | $z_d=\mathrm{lm\_head}(\mathrm{norm}(h_d))$，即官方 `hidden_states_list[d-1]` | Ouro 每轮末都过同一个 `model.norm`，下一轮输入就是归一化后的状态。第 2、4 轮用相同输出头和归一化，符合 plan |
| 精度 | 主干 bf16（模型原生精度）；`lm_head` 用 FP32 权重副本做 FP32 矩阵乘，关闭 TF32 | plan 要求截断与排名统一用 FP32。bf16 logits 在 15 附近的分辨率约 0.06，会制造大量假并列 |
| 参考输出 | $\arg\max z_4$（FP32）。greedy 轨迹就按它生成 | 参考 token 与生成 token 按构造一致，每个位置都会核验 |
| 中心化 | $z_d \leftarrow z_d-\mathrm{mean}_v z_d$，均值取全部 49,152 个词表行 | 按 plan。中心化不改变 $\arg\max$ 和 $m_2$，只影响 $z_4-z_2$ 的截断 |
| tie-breaking | 并列时取最小 token id（即 `torch.argmax` 的第一个最大值） | 全词表计算与紧凑重算一律使用此规则 |

前向在 `src/loopexp/ouro.py` 中重写，复用官方模块（embedding、24 层 sandwich-norm 层、`norm`、`lm_head`、RoPE）。每个（轮，层）有独立的 KV 槽，编号为轮 × 24 + 层，与官方 `UniversalTransformerCache` 一致。支持批量和左填充：填充位置不进入真实 query 的注意力，填充 query 只看自身，以免出现 NaN。

## 2. 数据、提示与轨迹

| 项 | 选择 |
|---|---|
| 数据集 | GSM8K test（1319 题，`openai/gsm8k` rev `740312a`）；MATH-500（500 题，`HuggingFaceH4/MATH-500` rev `6e4ed1a`）；MBPP+（378 题，`evalplus/mbppplus` rev `b2d74c9`） |
| 划分 | 按题划分：`sha256("loopexp-split-v1:<dataset>:<pid>") mod 10000 < 3000` 为校准集，约 30%；其余为测试集。同一回答的所有 token 归入同一划分 |
| 提示 | Ouro 自带的 ChatML 模板，零样本。数学题追加 “Please reason step by step, and put your final answer within \boxed{}.”；MBPP+ 给出题面和第一条测试，要求输出一个 ```python 代码块 |
| 解码 | greedy；`max_new_tokens=2048`；batch 8；遇 `<\|im_end\|>` 或 `<\|endoftext\|>` 停止 |
| 位置 | 每个生成的 token 计一个位置，从 prompt 后第一个 token 到停止 token（含）。达到长度上限的回答保留，并另做一组排除它们的稳健性检查 |
| KV | 解码全程使用原始四轮各自的 KV。所有方法只在读出端比较，前缀和缓存完全相同 |

提示格式的选择：loop_suffix 项目对 Ouro-1.4B 用的就是 chat 模板。冒烟测试（§8）中 chat 零样本输出正常：GSM8K 16/16 正确，MATH-500 12/16，MBPP+ 16/16 语法有效。第一阶段的准确率只用于确认轨迹有意义，不是结论指标。

长度上限的选择：Ouro-1.4B 在较难的 MATH 题上会切换成 “Okay, so I need to…” 式的长推理。上限为 1024 时冒烟中 7/16 被截断，2048 时降到 5/16；截断的都是正常推理，不是重复死循环。截断回答的每个位置仍是原始模型的真实 greedy 位置，所以只影响位置的分布（长回答的后半段被少采样），不影响逐位置结论。报告中单列截断率，并给出排除截断回答的稳健性检查。若要基本消除截断，可对 MATH 改用 4096，代价是 MATH 部分耗时约翻倍。

## 3. 每个位置保存什么（紧凑且可精确重算）

全词表 logits 无法全部保存（约 80 万位置 × 49k × 2，超过 100 GB）。对每个位置保存：

- $a_2=\arg\max z_2$、$a_4=\arg\max z_4$、生成 token、$m_2$、$m_4$；
- 候选集 $C=\mathrm{top}_k(z_2)\cup\mathrm{top}_k(z_4)$（$k=32$）上的 $z_2$、$z_4$ 值；
- **充分性证书**：每个 $\tilde z_{4,v}(B)$ 都落在 $z_{2,v}$ 与 $z_{4,v}$ 之间，所以赢家必须满足 $\max(z_{2,v},z_{4,v})\ge L:=\max_v\min(z_{2,v},z_{4,v})$。若 $z_2$、$z_4$ 的第 $k+1$ 大值都 $<L-10^{-3}$，候选集外的 token 对任何 $B\ge0$ 都不可能胜出或并列。没有证书的位置保存完整 FP32 $z_2$、$z_4$；
- 自检用：在 16 个固定 $B$（0 到 16）上用全词表直接算出的 $\arg\max\tilde z_4(B)$；
- 诊断量：$a_4$ 在 $z_2$ 中的排名，以及 $\mathrm{gap}=z_2(a_2)-z_2(a_4)$。

因此离线阶段能对**任意** $B$ 精确重算 $\arg\max\tilde z_4(B)$。校准出的 $B$ 不受预设网格限制。

## 4. 阈值校准

- 退出条件严格为 $m_2>2B$。
- **主方案（按数据集校准）**：在各数据集校准集的所有位置上取 $B_r=\tfrac12 q_{1-r}(m_2)$，$r\in\{10\%,25\%,50\%,75\%\}$。固定后在测试集报告实际退出率。
- **次方案（全局统一 $B$）**：每个 $r$ 选一个 $B$，使三个数据集校准退出率的等权平均为 $r$。用于检查跨任务共用一个 $B$ 的代价。
- **曲线**：主图不依赖校准。在测试集上按 $m_2$ 分位数扫 101 个 $B$，对应退出率 0%–100%。在同一 $B$ 下两种方法的退出集合相同，所以两条曲线横坐标一致。

## 5. 方法与指标

四种方法与 plan §4 完全一致。设 $\mathrm{deep}=[a_2\ne a_4]$（深层改判）、$\mathrm{exit}=[m_2>2B]$、$\hat a(B)=\arg\max\tilde z_4(B)$。

| 指标 | 定义（测试集，按 token 加权） |
|---|---|
| 可证明退出率 | $P(\mathrm{exit})$ |
| 退出位置分歧率 | $P(\mathrm{deep}\mid\mathrm{exit})$ |
| 截断额外代价 | $P(\neg\mathrm{exit}\wedge \hat a\ne a_4)$ |
| 深层改判保留率 | $P(\text{方法输出}=a_4\mid\mathrm{deep})$，有界修正与普通 early exit 各报一列 |
| 总分歧率 | 固定两轮 $P(\mathrm{deep})$；普通 early exit $P(\mathrm{exit}\wedge\mathrm{deep})$；有界修正 = 普通 early exit + 截断额外代价 |
| 损失来源（追加） | 深层改判中：因退出丢失的比例，与因截断在未退出位置丢失的比例。对应 plan §6 的“主要损失是否发生在已跑满四轮的位置” |
| 可达上界（追加） | 未退出的深层改判中 $\mathrm{gap}<2B$ 的比例。$\hat a=a_4$ 需要 $z_2(a_2)-z_2(a_4)<2B$，这是截断后仍保留改判的必要条件 |

置信区间：按题做 cluster bootstrap，1000 次重采样，取 2.5% 和 97.5% 分位数。同一回答内的 token 相关，所以重采样单位是题，不是 token。

**两条可推导的性质，可在实验中直接验证：**

1. $\tilde z_4(B)$ 对 $(z_2,z_4)$ 两个分量都单调不减。因此若 $a_2=a_4$，截断后输出不变（除非出现 FP32 完全相等的并列）。截断只可能在**深层改判位置**造成损失，所以截断额外代价 ≤ $P(\neg\mathrm{exit}\wedge\mathrm{deep})$。
2. 由第 1 条可得，有界修正的总分歧率恰好等于 $P(\mathrm{deep})\times(1-\text{保留率})$。“总分歧率低”与“多数深层改判被保留”是同一件事的两种尺度，需要分别给出的是 $P(\mathrm{deep})$ 这个基数。

## 6. 实现自检（必须全部通过，结果才有效）

| 检查 | 通过标准 | 位置 |
|---|---|---|
| 单元测试：退出保证、$B=0$ 与 $B\to\infty$ 两端、一致位置不变、紧凑与全词表一致、tie-breaking | `pytest` 全绿 | `tests/test_bounded.py` |
| 合成数据端到端：保存格式到分析；恒等式“有界修正分歧 = 普通 early exit 分歧 + 截断代价”，以及“分歧 = $P(\mathrm{deep})(1-\text{保留率})$” | 全绿 | `tests/test_pipeline.py` |
| 重写前向 vs 官方前向（teacher-forcing、无缓存、单次并行） | 第 2、4 轮 argmax 一致率 ≥ 99%（bf16 kernel 差异只影响近似并列的位置）；同时报告最大 $\lvert\Delta z\rvert$ 和 $\lvert\Delta m_2\rvert$ | `scripts/validate_forward.py` |
| 批量解码（6 题一批）vs batch=1 重放同一续写 | 同上 | 同上 |
| 每个位置：生成 token = $\arg\max z_4$ | 100% | `collect.py` 写盘时与 `analyze.py` 中检查 |
| 紧凑重算 vs 全词表直接计算（16 个 $B$ × 全部位置） | 0 个不一致 | `analyze.py` |
| 满足 $m_2>2B$ 的位置 $\hat a=a_2$（网格 $B$、校准 $B$ 与曲线 $B$） | 0 例违反 | `analyze.py` |
| $a_2=a_4$ 的位置截断后不变 | 0 例（如有，应是 FP32 并列，需逐例说明） | `analyze.py` |

## 7. 是否继续：建议预注册的判定（**待你确认后再跑全量**）

plan §6 的判定是定性的。建议在看到测试集结果前固定下面的数值版本：

- **值得继续**：三个数据集中，至少数学两个数据集存在一个校准 $B$，使测试退出率 ≥ 25%（理想 loop 计算节省 ≥ 12.5%），**且**有界修正保留率 ≥ 50%，**且**截断导致的损失 ≤ 退出导致的损失。满足后，在该区间选 1–2 个 $B$ 进入第二阶段（自由生成，测数学准确率与 MBPP+ pass@1）。
- **不值得继续固定阈值方案**：对所有退出率 ≥ 25% 的 $B$，保留率 < 50%；或者截断损失 > 退出损失，即主要损失出现在已跑满四轮的位置。
- 其余情况视为不确定，按数据集分别讨论。可证明退出率**不**当作加速比（plan §6）。

## 8. 已完成的验证（2026-11-11）

| 项 | 结果 |
|---|---|
| `pytest`（单元测试与合成端到端，共 6 项） | 全部通过 |
| 重写前向 vs 官方前向（6 题、474 个位置） | 第 2 轮 argmax 一致率 99.37%，第 4 轮 99.79%；$\lvert\Delta z\rvert$ 均值 0.03–0.07，最大 5.4；$\lvert\Delta m_2\rvert$ 中位数约 0.05，最大 1.4。通过（`results/validate_forward.json`） |
| 批量（6 题一批）vs batch 1 重放 | 第 2 轮 99.37%，第 4 轮 100%；最大 $\lvert\Delta m_2\rvert$ 1.6。通过 |
| 冒烟数据（每个数据集 16 道校准题，共 19,609 个位置）上的自检 | 证书覆盖率 99.78%–99.87%；生成 token = $\arg\max z_4$ 100%；紧凑与全词表不一致 0；退出保证违反 0；一致位置被改变 0 |
| 速度（共享 GPU，对方作业占用率接近 100%） | 上限 2048、batch 8：GSM8K 约 12 秒/题，MATH-500 约 17 秒/题，MBPP+ 不到 1 秒/题 |
| 存储 | 约 1.8 KB/位置（含未获证书位置的全词表行） |

**数值噪声提示。** 同一个模型换一种 kernel 或 batch 形状，个别位置的 $m_2$ 会变化到约 1.5 个 logit。本实验的保证是针对**本实现实际算出的** $z_2,z_4$，在这个意义上是严格的（自检为 0 违反）。部署到数值不同的 kernel 时，退出条件需要留余量，即 $m_2>2B+\delta$。第二阶段若换推理后端，需要先测出 $\delta$。

冒烟数据只有 16 题，并且是在校准集上自评，曲线不能用来下结论。全量跑完才看测试集结果。

## 9. 执行步骤与资源

```bash
cd /mnt/disk2_from_server2/siqizhu4/LoopExperiments
PY=.venv/bin/python
# 0. 单元与端到端测试（CPU）
$PY -m pytest -q
# 1. 前向等价性（单卡，约 2 分钟）
$PY scripts/validate_forward.py --device cuda:0
# 2. 冒烟：每个数据集取 16 道校准题，检查准确率、输出、速度与证书率
for ds in gsm8k math500 mbppplus; do $PY scripts/collect.py --dataset $ds --split calib --limit 16 --out runs/smoke --device cuda:0; done
$PY scripts/analyze.py --run runs/smoke
# 3. 全量（每张卡一个进程，按 shard 切；可断点续跑；OOM 时自动把 batch 对半拆开重试）
bash scripts/run_full.sh runs/v1 "<gpu 列表>"
$PY scripts/analyze.py --run runs/v1 --device cuda:0
```

全量规模：2,197 题，约 120 万个位置。按冒烟速度估计约 7 GPU·小时（GSM8K 约 4.4，MATH-500 约 2.3，MBPP+ 不到 0.2），4 张卡约 2 小时。每卡显存峰值约 15–20 GB。产出约 2 GB，写在 `runs/v1/`。服务器有空闲 GPU 回收器，本作业持续占用 GPU，不受影响。

产物：`runs/v1/analysis/report.md`（自检、两种校准表、损失来源、稳健性检查、轨迹统计）、`main.pdf/png`（上排为分歧率 vs 退出率，下排为保留率 vs 退出率，每个数据集一列）、`m2_hist.pdf`、`results.json`、`curves.json`。

## 10. 不在本阶段

- 自由生成下的准确率和 pass@1（第二阶段）。MBPP+ 的代码执行需要沙箱，本阶段只统计语法有效率。
- 真实加速：需要另行考虑后续 token 对深层 KV 的补算（plan §6）。
- 其他模型（Ouro-2.6B 等）和逐位置自适应的 $B$。
