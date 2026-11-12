# 方向一核验：Loop 所需轮数由总步数决定，还是由最长依赖链决定？

状态（2026-11-12，sn4622128200，GPU 2）：先导实验已完成，结果见 [RESULTS.md](RESULTS.md)。

## 1. 把假说拆成可检验的命题

| 编号 | 命题 | 可证伪的预测 |
|---|---|---|
| C1 行为 | 所需 loop 数随最长依赖链 D 增长，基本不随总操作数 N 增长 | 固定 N 时，两分支程序（D 小）在每一轮的准确率都不低于串行链（D=N）。固定 D 时，给第二条分支加操作，准确率基本不变 |
| C2 机制 | 两条独立分支的中间结果在同一轮形成，并且可以分别使用 | 在第 r 轮把两个来源程序的分支状态同时换入，输出变成组合答案（A'∘B'）。只换一支时分别得到 A'∘B 和 A∘B'。不同写法、相同分支值的来源给出相同结果 |
| C3 训练 | 按依赖图监督每轮可完成的步骤，优于线性 CoT 监督和随机拓扑序 | 只有 C1、C2 成立时才做 |

文献核查（2026-11-12，后台检索）：C1 的理论上界已有（Xu & Sato, ICML 2026：looped/latent thought 用 O(depth) 轮可计算 DAG，CoT 需要 O(size) 步）。串行链“每轮约一跳”已在小型 looped 模型上用逐轮 patching 证实（Sesodia et al., NeurIPS 2026, *Inside the Loop*），但只测串行链。在预训练 Ouro/Huginn 上，**固定操作数比较串行与并行**、以及**双来源组合替换**，没有找到先例。按依赖图做逐轮监督也没有找到先例。主要风险：Ouro 多数答案在第 1 轮就定下；中间值可读但不被使用（Merin 2026）；Ouro 每轮重建状态，单轮 patch 可能被覆盖。

## 2. 任务设计（`progs.py`）

单字母变量的 SSA 程序，所有值都是 0–9，答案是一个数字 token。两分支族 TB(L1, L2)：

```
a = A0            # 分支 1 的输入
b = B0            # 分支 2 的输入
p1 = a ± c ...    # 分支 1：L1 个“变量 ± 常数”
q1 = b ± c ...    # 分支 2：L2 个“变量 ± 常数”
m = pL1 ± qL2     # 唯一一个“变量 ± 变量”的合并
print(m)
```

- N = L1 + L2 + 1，D = max(L1, L2) + 1。L2 = 0 就是串行链（b 在最后并入）。
- **固定 N 时，所有拆分的行数、token 数、操作类型计数完全相同**，只有依赖结构不同。“var-lit”一行 7 个 token，“var-var”一行 6 个 token。
- 固定 D（固定 L1）时增加 L2，就是“加必须参与答案的并行操作”。
- 每个格子内答案在 0–9 上均匀（拒绝采样），变量名随机。行序分 grouped（先分支 1 后分支 2）和 interleaved（交替）。
- 提示：4 个固定 few-shot 例子（浅层形状），以 `Output: ` 结尾。数字的分词是“空格 + 数字”，所以下一个 token 就是答案。

## 3. 读取与干预（`loopmodel.py`）

- 用官方 decoder layer 模块做一次无缓存并行前向。Ouro 从第 r 轮传到第 r+1 轮的唯一状态是各位置归一化后的隐状态 h_r（每轮有独立的 KV，由本轮输入算出），所以 h_r 是完整的干预点。h_0 是 embedding。
- 第 r 轮的读出 = lm_head(h_r)，即 Ouro 训练过的逐轮退出预测。跑 R 轮读第 r 轮等价于只跑 r 轮。
- 自检：与官方 `OuroModel` 前向逐 logit 比较，1.4B 和 2.6B 的最大差都是 0.0。左填充批量与单条相比最大差 0.2 logit（bf16 kernel 形状差异）。

## 4. 实验

**Exp 1（C1）**，`exp1_loops.py` + `analyze_exp1.py`：所有 L1 ≥ L2、N ≤ Nmax 的格子，两种行序，读第 1–8 轮的 10 个数字分布（Ouro 训练 4 轮，5–8 轮是外推）。主表是第 r 轮准确率按 L1 × L2 排列，以及固定 N 下“两分支 − 串行链”的差和 95% 置信区间。

**Exp 2（C2）**，`exp2_patch.py` + `analyze_exp2.py`：TB(1,1) 基底 `a=A0; b=B0; p=a±c1; q=b±c2; m=p±q`。
- 来源 1 同时改 A0 和 c1（都与基底不同），得到 p = A'。来源 2 同理改分支 2。每个目标值用 3 条不同路线（不同的 A0'、运算、c1'）。
- 在第 r 个边界把来源状态复制到基底的若干位置，继续跑到第 4 轮。位置集合有三种：
  - `line`：分支运算行中与基底 token 相同的位置。
  - `branch`：分支两行中与基底 token 相同的位置。
  - `full`：分支两行全部位置（r=0 时等于直接换程序文本，作阳性对照）。
- 因为 `line` 和 `branch` 不复制不同的数字 token，r=0 时严格无效（自检：差为 0）。任何效果都只能来自计算出的状态，排除了“搬运输入数字”。
- 两种模式：once（只在边界 r）和 onward（r 及之后每个边界都换）。三种替换：A、B、AB。
- 还记录“操作数搬运”的替代预测 (A0' op1 c1)∘B，并检查不同路线给出的答案是否一致。

**Exp 2b（定位）**，`exp2b_trace.py` + `analyze_exp2b.py`：逐个位置、逐个边界单点替换，画出“答案跟随来源”的概率热图，找出分支值在哪个位置、哪一轮存在并被下游使用。

## 5. 判定（沿用原提案的去留规则）

- C1 被支持：固定 N 时两分支 ≥ 串行链（置信区间不含负值），且固定 D 时 L2 的斜率接近 0。
- C1 被否定：固定 N 时串行链显著更好，或者 L2 的代价与 L1 相当甚至更大。这时按原提案“停在诊断结果”，不做 C3。
- C2：要求单替换分别得到 A'∘B 和 A∘B'，同时替换得到 A'∘B'，且不同路线结果一致。之后才谈“最早在哪一轮同时出现”。

## 6. 复现

```bash
cd /mnt/disk2_from_server2/siqizhu4/LoopExperiments/dag_depth
PY=../.venv/bin/python   # 环境见上级 README；本机 2026-11-12 重建
CUDA_VISIBLE_DEVICES=2 $PY exp1_loops.py --n 100 --max-n 9 --out runs/pilot1                        # 1.4B
CUDA_VISIBLE_DEVICES=2 $PY exp1_loops.py --model ../../models/ouro-2.6b --n 300 --max-n 6 --seed 7 --out runs/exp1_2p6b
CUDA_VISIBLE_DEVICES=2 $PY exp2_patch.py --n 400 --routes 3 --out runs/exp2_2p6b
CUDA_VISIBLE_DEVICES=2 $PY exp2b_trace.py --n 200 --out runs/exp2b_2p6b
$PY analyze_exp1.py --run runs/exp1_2p6b; $PY analyze_exp2.py --run runs/exp2_2p6b; $PY analyze_exp2b.py --run runs/exp2b_2p6b
```

Ouro-2.6B 权重：`/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b`（`ByteDance/Ouro-2.6B` rev `1ed0425`，2026-11-12 下载）。
