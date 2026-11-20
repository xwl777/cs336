# FlashAttention-2 Triton Backward：Algorithm 2 详细解读

本文解释作业 4.2.3 OPTIONAL 中的分块反向算法，以及如何把它映射到 Triton program、launch grid 和局部矩阵运算。实现现已整合到 `flash_attention.py`。

相关材料：

- [当前实现](cs336_systems/flash_attention.py)
- [前向算法与代码说明](FLASH_ATTENTION_FORWARD_GUIDE.md)
- [正确性测试](tests/test_attention.py)
- [benchmark 说明](benchmark/FLASH_BENCHMARK.md)

第 1–17 节解释算法和实现设计；第 18–25 节逐段解读当前代码。初次阅读建议先读第 3–6 节，再对照第 18 节之后的代码说明。

## 1. 先明确当前实现与这道可选题的区别

当前 `FlashAttentionTriton` 的前向和反向均使用 Triton。原有 `_flash_backward_pytorch` 是用 `torch.compile` 编译的 PyTorch 函数，保留给 `FlashAttentionPytorch` 使用。

原有 PyTorch 反向虽然没有从前向保存完整 P，却会在反向重算完整 S、P、dP、dS，仍有平方级中间张量内存。

可选题要求把反向也分块：只在 GPU 芯片上生成当前块的 S/P/dP/dS，累加当前输出块，最后只写回 dQ、dK、dV。它不要求改变 attention 的数学定义。

关键设计是：**分别按 key 块计算 dK/dV，按 query 块计算 dQ。两组计算各自重算 P，从而让每个最终梯度块只有一个写入者。**

## 2. 输入、输出和符号

先只考虑一个 batch，没有 head 维度：

```text
Q、O、dO: (Nq, d)
K、V:     (Nk, d)
L:        (Nq,)

输出：
dQ: (Nq, d)
dK: (Nk, d)
dV: (Nk, d)
```

加入 batch 后，所有形状最前面增加 B。例如 Q 为 `(B, Nq, d)`，L 为 `(B, Nq)`。不同 batch 的 attention 和梯度彼此独立。

符号含义：

- `dO`：上游传来的损失对输出 O 的梯度，不是本 kernel 要求出来的输出。
- `dQ/dK/dV`：本次 backward 要返回的输入梯度。
- `L`：前向保存的每行 logsumexp；它不是损失。
- `D`：本次 backward 预计算的每行一个数的向量；它不是 embedding 维度 d。
- `Bq/Bk`：query/key 的块大小。
- `Tq = ceil(Nq/Bq)`、`Tk = ceil(Nk/Bk)`：query/key 的块数。

Algorithm 2 用下标 i 表示 query 块，上标 (j) 表示 key 块。例如 P_i^(j) 是第 i 个 query 块和第 j 个 key 块相遇时产生的概率块。

## 3. 反向公式从哪里来？

前向为：

$$
S=QK^\top/\sqrt d,\qquad P=\operatorname{softmax}(S),\qquad O=PV.
$$

用损失对 O 的梯度 dO，依次得到：

$$
dV=P^\top dO,\qquad dP=dO V^\top.
$$

softmax 的反向为：

$$
dS_{ij}=P_{ij}\left(dP_{ij}-\sum_tP_{it}dP_{it}\right).
$$

定义：

$$
D_i=\sum_tP_{it}dP_{it}.
$$

则：

$$
dS=P\circ(dP-D),
$$

其中 D 沿 key 维广播，圆圈表示逐元素乘法。最后：

$$
dQ=dS K/\sqrt d,\qquad dK=dS^\top Q/\sqrt d.
$$

注意 dS 是相对于**已经缩放的 S** 的梯度，因此在求 dQ/dK 时还需要乘一次 `scale = 1/sqrt(d)`。dV 不需要乘 scale。

## 4. 为什么 D 可以提前计算？

直接用 `D = rowsum(P * dP)`，似乎需要先重建完整 P 和 dP。利用 O = PV，可以避开它们：

$$
\begin{aligned}
D_i
&=\sum_jP_{ij}dP_{ij}\\
&=\sum_jP_{ij}\sum_t dO_{it}V_{jt}\\
&=\sum_t dO_{it}\sum_jP_{ij}V_{jt}\\
&=\sum_t dO_{it}O_{it}.
\end{aligned}
$$

因此预处理只需沿 embedding 维求和：

$$
\boxed{D=\operatorname{rowsum}(O\circ dO)}.
$$

O/dO 的形状为 `(Nq, d)`，D 为 `(Nq,)`。包括 batch 时，D 为 `(B, Nq)`。

这一步只有线性规模的数据访问。D 写到显存后，可供两个反向 kernel 重复读取。

Algorithm 2 还描述了对 O 分块，但 O 在预处理完 D 之后就不需要再被主反向循环读取；主循环读取的是 Q、K、V、dO、L、D。

## 5. 为什么用 L 就能重算 P？

前向保存：

$$
L_i=\log\sum_j\exp(S_{ij}).
$$

因此对任意一个 score 块：

$$
P_{ij}=\exp(S_{ij}-L_i).
$$

这里 L 已包含整行全部 key 的归一化信息，所以反向拿到一个局部 S 块后，就能直接得到该块对应的最终概率。

**反向不需要重新做 online softmax，不需要追踪 m/l，也不能只对当前块独立做 softmax。** 局部 softmax 的分母只包含当前块，会改变 attention 的定义。

## 6. 为什么要把 dQ 和 dK/dV 分开算？

把梯度公式按块展开：

$$
dQ_i=\sum_j dS_i^{(j)}K^{(j)}/\sqrt d,
$$

$$
dK^{(j)}=\sum_i(dS_i^{(j)})^\top Q_i/\sqrt d,
$$

$$
dV^{(j)}=\sum_i(P_i^{(j)})^\top dO_i.
$$

观察求和方向：dQ_i 需要遍历所有 key 块 j；dK^(j)/dV^(j) 需要遍历所有 query 块 i。

假设像前向一样，每个 program 负责一个 query 块：

- 它能遍历全部 key，把自己的 dQ_i 完整算出来。
- 但对每个 dK/dV 块，它只能给出来自当前 query 块的一部分贡献。
- 其他 query program 也会向相同 dK/dV 地址写入自己的贡献。

普通 store 会覆盖，普通“读取旧值再相加写回”会产生并发竞争。为了正确求和，需要原子加法，或者额外的部分结果缓冲区和归约步骤。

Algorithm 2 选择另一条路径：给每个 key 块单独分配一个 program，让它自己遍历所有 query，完整计算该 key 块的 dK/dV。然后另起一组 query program 完整计算 dQ。

代价是两组 program 都要重算 S/P/dP/dS；收益是所有梯度块都只有一个写入者，不需要原子累加，也不需要平方规模的部分梯度缓冲区。

### 6.1 两个 query 块、两个 key 块的例子

局部 score/probability 块排列为：

```text
              key 0       key 1
query 0       P_0^(0)     P_0^(1)
query 1       P_1^(0)     P_1^(1)
```

计算 dK/dV 的第一组 program：

- key-program 0 处理左列两个块，只写 dK_0、dV_0。
- key-program 1 处理右列两个块，只写 dK_1、dV_1。

计算 dQ 的第二组 program：

- query-program 0 处理上行两个块，只写 dQ_0。
- query-program 1 处理下行两个块，只写 dQ_1。

每个 P 块在两组计算中各产生一次，用完即丢弃，不写入完整 P 矩阵。

## 7. Triton 实现可以拆成三个 kernel

一种直接对应 Algorithm 2 的组织方式是：

1. 预处理 kernel：计算 D。
2. dK/dV kernel：一个 program 负责一个 key 块。
3. dQ kernel：一个 program 负责一个 query 块。

这是实现组织建议，不是算法唯一允许的 kernel 划分。

### 7.1 预处理 D 的 grid

可以使用二维 grid `(ceil(Nq / Br), B)`，Br 是每个 program 处理的 query 行数。

`tl.program_id(0)` 选择行块，`tl.program_id(1)` 选择 batch。每个 program 加载 `(Br, d)` 的 O/dO，转换为 float32 后逐元素乘，再沿 embedding 维归约，写回 `(Br,)` 的 D。

这里的归约轴是 `axis=1`，因为加载的 tensor 是二维 `(行, embedding)`。D 的输出 buffer 建议为 float32。

### 7.2 kernel 之间是否需要 synchronize？

三个 kernel 在同一个 CUDA stream 中依次启动时，stream 顺序保证后面的 kernel 可以读取前面写好的 D。通常不需要在每次启动后调用主机端 `torch.cuda.synchronize()`。

“避免 synchronization”主要指避免主循环中不同 program 之间协调梯度累加，不是说整个实现没有任何数据依赖。普通 block 内的 barrier 不能解决不同 program 的并发写入问题。

## 8. dK/dV kernel：对应第 6–19 行

### 8.1 grid 和固定任务

grid 取 `(Tk, B)`：第 0 维选择 key 块，第 1 维选择 batch。

对某个 program，j 固定，i 在循环中变化。这个 program 最终负责写完整的 `(Bk, d)` 的 dK^(j) 和 dV^(j)。

### 8.2 哪些数据只加载一次？

K^(j)、V^(j) 在 query 循环外加载，形状均为 `(Bk, d)`。因为所有 query 块都要与当前固定的 key 块相互作用，它们可以被重复使用。

初始化两个 float32 累加器：dK_acc、dV_acc，形状均为 `(Bk, d)`。

### 8.3 第一次进入循环时加载什么？

加载当前 Q_i、dO_i、L_i、D_i：

```text
Q_i、dO_i: (Bq, d)
L_i、D_i:  (Bq,)
```

对当前的 query/key 块，计算 score：

```text
Q_i:                  (Bq, d)
transpose(K_j):       (d, Bk)
S = dot(Q_i, K_j^T):  (Bq, Bk)
```

乘 scale 后，应用与前向一致的 causal mask，再用 L 重算概率。

### 8.4 L 和 D 为什么需要增加一维？

S/P/dP 的形状为 `(Bq, Bk)`，而 L/D 的形状为 `(Bq,)`。要让同一 query 行使用同一个 L/D，需要把它们看成列向量：

```text
L[:, None]、D[:, None]: (Bq, 1)
```

这样与 `(Bq, Bk)` 广播时，每行的值沿 key 维扩展，得到：

$$
P=\exp(S-L[:,None]),\qquad dS=P\circ(dP-D[:,None]).
$$

如果写成 `D[None, :]`，会沿错误的维度广播；Bq 恰好等于 Bk 时甚至可能不报 shape 错误，但数学含义已经错了。

### 8.5 累加 dV

$$
dV_{acc}\mathrel{+}=P^\top dO_i.
$$

形状：`(Bk, Bq) × (Bq, d) -> (Bk, d)`。左边要转置的是 P，不是 dO。这里沿 query 维归约，得到当前 query 块对固定 key 块的贡献。

### 8.6 求 dP 和 dS

$$
dP=dO_i(V^{(j)})^\top.
$$

形状：`(Bq, d) × (d, Bk) -> (Bq, Bk)`。再逐元素执行 softmax 反向，得到同形状的 dS。

### 8.7 累加 dK

$$
dK_{acc}\mathrel{+}=dS^\top Q_i/\sqrt d.
$$

形状：`(Bk, Bq) × (Bq, d) -> (Bk, d)`。scale 可以放到每次贡献中，也可以在 float32 累加全部贡献后统一乘一次；不能两处都乘。

### 8.8 推进哪些指针？

循环中变化的是 query 块，因此推进 Q、dO、L、D 的 block pointer，并增加 query 的全局索引：

- Q/dO 在行维移动 Bq，embedding 维不动。
- L/D 在唯一的行维移动 Bq。
- K/V 和最终 dK/dV 的输出位置保持不变。

循环结束后，将两个 float32 累加器转换为梯度输出 dtype，各自 store 一次。其他 key-program 不会写这些地址。

## 9. dQ kernel：对应第 20–32 行

### 9.1 grid 和固定任务

grid 取 `(Tq, B)`。每个 program 固定一个 query 块 i，最终写完整的 dQ_i。

循环外加载 Q_i、dO_i、L_i、D_i，初始化 `(Bq, d)` 的 float32 dQ_acc。它们在所有 key 轮次中保持不变。

### 9.2 循环中执行什么？

每轮加载 K^(j)、V^(j)，重算与上一组 kernel 相同的局部 S、P、dP、dS。然后：

$$
dQ_{acc}\mathrel{+}=dS K^{(j)}/\sqrt d.
$$

形状：`(Bq, Bk) × (Bk, d) -> (Bq, d)`。这次沿 key 维归约，因此不转置 dS。

### 9.3 推进哪些指针？

循环里推进 K/V 的 block pointer，增加 key 的全局索引；Q/dO/L/D 不动。循环结束后将 dQ_acc 转为输出 dtype，写回当前 query 行。

这组 kernel 不计算 dK/dV，因此它不会与其他 query-program 争抢 dK/dV 的写入位置。

## 10. block pointer 应如何理解？

三维张量的 batch 偏移先由 `base + batch_index * stride_batch` 处理。进入某个 batch 后，再用二维 block pointer 描述 `(序列, embedding)`。

对于连续布局，通常行 stride 是 d，embedding stride 是 1，`order=(1, 0)`。shape 描述当前 batch 的完整矩阵，offsets 描述当前块的起点，block_shape 描述本轮的块大小。

dK/dV kernel 的初始 K/V offset 是 `(key_tile_index * Bk, 0)`；dQ kernel 的初始 Q/dO offset 是 `(query_tile_index * Bq, 0)`。块索引必须乘块大小。

`tl.trans` 转置的是已经加载的局部 tensor。如果 block pointer 本身已经按转置视图构建，后续又转置一次，会把形状和公式弄错。先采用统一的 `(序列, embedding)` 指针布局更容易检查。

## 11. causal mask 在两个 kernel 中都要恢复

不论当前是按 query 固定还是按 key 固定，最终 P 块的行永远对应 query、列对应 key。

通过全局索引构造允许条件：

$$
\operatorname{allowed}_{ab}=(q_{global,a}\ge k_{global,b}).
$$

query 全局索引的形状为 `(Bq,)`，key 全局索引的形状为 `(Bk,)`，增加维度后比较得到 `(Bq, Bk)`。

前向遮掉的连接在反向也必须遮掉。只在前向 mask、反向却用未 mask 的 score 重算 P，会产生错误概率和梯度。

当前代码前向对屏蔽位置加 -1e6。以它为基准重算时，应保持遮罩约定；通常这些位置的指数在 float32 中下溢为零。若采用严格的负无穷遮罩，则其概率严格为零。两者在任意输入范围下不完全等价，不能无条件混用。

### 11.1 能否跳过整个 causal 无效块？

对于标准索引规则 k ≤ q，如果一个 key 块的最小索引都大于当前 query 块的最大索引，则整个块没有有效连接，可以跳过其计算。对跨越对角线的块，仍需逐元素 mask。

dQ kernel 固定 query 块，可以只遍历可能有效的 key 块；dK/dV kernel 固定 key 块，可以从可能有效的 query 块开始遍历。

应先实现完整遍历并验证，再缩小循环范围。对于有限 -1e6 遮罩，跳过块是在采用“被屏蔽概率为零”的数值约定下做的优化。

## 12. dtype 和数值精度的实现要点

Q/K/V/dO 可以是 BF16 或 FP32，但 D 的归约、局部概率及梯度公式、dQ/dK/dV 累加器应优先使用 float32，避免长序列累计误差。

BF16 输入的 `tl.dot` 可以使用 BF16 乘法和 float32 累加。对 P^T dO、dS^T Q、dS K 等乘法，P/dS 通常要先转换为另一输入的 dtype，dot 的累加器仍保持 float32。

不要每轮把梯度累加器转回 BF16，否则会反复舍入；在最终 store 前再转为输出 dtype。使用 `tl.dot` 的 acc 参数可以直接累加局部贡献。

FP32 dot 的输入精度选项也影响误差和性能。当前前向使用 `tf32x3`，反向应明确选择并记录精度策略；不能默认所有 FP32 GPU 路径都逐操作等同于 IEEE FP32。

当前 L 保存的是自然对数 logsumexp，所以重算 P 使用自然指数 exp。如果换成 exp2，需要同步处理 `log2(e)` 的换底因子；直接使用 `exp2(S-L)` 会算错。

## 13. 越界处理和特殊输入

题目保证测试维度是至少 16 的 2 的幂。选择能整除对应长度的 tile，可先省去复杂尾块处理。

如果扩展到一般长度，仅给越界 K/V 补零不够：补零 K 会产生 score=0，若不屏蔽，该位置仍可能贡献 softmax 概率。必须结合 query/key 有效索引 mask，并对无效输出行禁止 store。

对无效 query 行加载的 L/D 也要小心，避免 `exp` 或 `inf-inf` 产生 NaN 后污染有效的 dK/dV 累加。需要在计算路径中确保无效行的 P/dS 贡献为零。

如果 dO 不连续，不能直接沿用 O 的 strides；它有自己的实际布局。可以让指针显式接收各张量 strides，或者在包装函数中转连续，但后一种选择可能引入拷贝开销。

## 14. autograd.Function.backward 如何组织？

forward 仍保存 Q、K、V、O、L 和 causal 标志。backward 收到 dO 后：

1. 取回保存的数据与 `ctx.is_causal`。
2. 分配 float32 的 D，以及与输入形状/dtype 对应的 dQ/dK/dV。
3. 启动 D 预处理 kernel。
4. 启动 dK/dV kernel，grid 为 `(Tk, B)`。
5. 启动 dQ kernel，grid 为 `(Tq, B)`。
6. 返回与 forward 输入一一对应的梯度。

forward 显式输入为 Q、K、V、is_causal 时，梯度返回值为 dQ、dK、dV、None；布尔值不需要梯度，ctx 不占梯度返回位置。

如果使用 `torch.empty` 分配梯度，必须保证所有有效元素都被完整写入。这种输出独占设计允许每个 program 从局部零累加器出发，最后一次 store，不需要把整个梯度输出先清零。

## 15. 它节省了多少内存？计算量会下降吗？

当 Nq = Nk = N 时，普通反向可能创建 B × N × N 的 S/P/dP/dS。一个 float32 矩阵就占 `4*B*N²` 字节。

B=1、N=65536 时，仅一个完整矩阵就占 16 GiB；多个矩阵同时存活会造成很高的显存峰值。

分块反向在显存中保留 Q/K/V/O/dO/L，增加 D 和三个输入梯度，不创建完整 attention 中间矩阵。d 固定时，这些主要 buffer 随 N 线性增长。

单个 program 的局部临时状态主要是 Bq × Bk 的 score/概率/梯度块，以及 Bq × d 或 Bk × d 的梯度累加器。实际寄存器、shared memory 和是否 spill，取决于编译器布局、tile 大小与硬件，不能把所有局部 tensor 的逻辑大小直接当作精确 SRAM 占用。

两组 kernel 重算部分相同数据，增加了算术工作。稠密 attention 的总计算复杂度仍是 O(B*Nq*Nk*d)。性能收益来自减少 HBM 读写、大中间张量、跨 program 的累加协调；重复计算换来的收益要由目标 GPU 上的 benchmark 验证。

## 16. 如何测试与定位问题？

在支持 Triton 的 CUDA 机器上运行：

```bash
uv run pytest -k test_flash_backward
```

原有 Triton backward 测试覆盖 causal=False/True，主要使用 FP32，并将 dQ/dK/dV 与普通 PyTorch attention 的 autograd 梯度比较。新增 `test_flash_backward_triton_tiled` 覆盖 BF16、不同 d、尾块和不同 query/key 长度；较长序列与更多 tile 组合仍需在 GPU 上验证。

建议按依赖顺序检查：

1. D 是否与 `(O.float() * dO.float()).sum(-1)` 一致。
2. 某个局部 S/P 是否与参考矩阵对应切片一致。
3. 局部 dP/dS 是否正确，D 是否沿 key 维广播。
4. 先检查 dV，再检查 dK，最后检查 dQ。
5. 检查每个输出块是否被唯一 program 完整写入。

典型错误的排查方向：

- dV 正确但 dQ/dK 错误：优先检查 D、dS 和 scale。
- 非 causal 正确，causal 错误：检查全局索引、mask 和循环范围。
- 只有 BF16 报错：检查 dot 输入 dtype 是否匹配。
- 长序列误差增大：检查是否使用 float32 累加，以及 dot 精度。
- 改 tile 后结果改变很大：检查 offset 是否乘 tile 大小、advance 轴和越界处理。
- 错误随运行变化：检查多个 program 是否写了相同地址，或输出是否有未初始化元素。

测试通过后再计时。benchmark 的 backward 必须调用新的 Triton 路径，不能仍然调用当前的 compiled PyTorch helper。两条路径的显存行为与性能不同，结果表和元数据应明确记录实际使用的反向实现。

## 17. 对照 Algorithm 2 阅读

- 第 2 行：预计算 D；它用 O/dO 代替整行 P/dP 的归约。
- 第 3–5 行：描述切块和索引，不一定需要在 Python 中真的创建所有切片对象。
- 第 6–19 行：并行分配 key 块，在 program 内顺序遍历 query 块，完整计算 dK/dV。
- 第 20–32 行：并行分配 query 块，在 program 内顺序遍历 key 块，完整计算 dQ。
- 第 33 行：将三个梯度交回 autograd 引擎。

每次阅读或实现一个局部运算时，都可以问两个问题：这个结果沿 query 还是 key 求和？最终由哪个 program 独占写入？这两点决定了矩阵是否转置、循环方向和是否会发生写入竞争。

当前代码映射：`flash_bwd_preprocess_kernel` 计算 D，`flash_bwd_dkdv_kernel` 计算 dK/dV，`flash_bwd_dq_kernel` 计算 dQ；`_flash_bwd_prob_and_ds` 复用局部概率和 softmax 梯度公式。三个反向 kernel 均使用 `tl.make_block_ptr` 读写，主循环通过 `.advance()` 移动指针。

`BWD_Q_TILE_SIZE/BWD_K_TILE_SIZE` 默认 32/64，forward 会保存它们供对应 backward 使用。当前包装类接受 `(B, N, d)`，没有添加 head 维度。新增的 GPU 测试位于 `tests/test_flash_triton_backward.py`，覆盖 FP32/BF16、causal/non-causal、尾块、不同 query/key 长度和非连续上游梯度；本地未运行 GPU 测试，不能视为已在 CUDA 验证。

## 18. 当前代码的命名与寻址约定

后面的代码片段均取自当前 [flash_attention.py](cs336_systems/flash_attention.py)，便于一边打开源文件一边阅读。

先区分两个容易混淆的名字：kernel 参数 `D` 表示 embedding 维度 d；`Delta_ptr` 指向算法第 2 行的 D 向量。局部变量 `delta` 也是算法中的 D，不是一个新的数学量。

`DO_ptr` 指向上游梯度 dO；小写 `do` 是加载到 program 内的 dO tile。这里 dO 只是梯度符号，并非“对 O 再执行一次微分操作”。

所有 kernel 都定义在 `if triton is not None` 内，未安装 Triton 时不会定义或编译它们。`@triton.jit` 让 Triton 在实际使用时为给定签名编译，而不是在导入模块时执行全部 GPU 运算。

### 18.1 为什么没有给反向 kernel 传 strides？

forward 在保存前把 Q/K/V 转为连续张量，O/L 也按连续布局分配。backward 把收到的 dO 转连续，因此反向内部采用固定的连续布局地址公式。

对 `(B, N, d)` 张量，元素 `(b, r, c)` 的元素偏移是：

$$
\operatorname{offset}(b,r,c)=bNd+rd+c=(bN+r)d+c.
$$

当前不再手动生成每个元素的地址矩阵，而是通过 `tl.make_block_ptr` 描述内存中的块。block pointer 内部根据 base、strides 和 offsets 定位元素；`boundary_check` 负责读取和写入的边界检查。

两种方式都能实现 Algorithm 2。必须保证地址公式与真实布局相符；不能把这些 kernel 单独拿去接收任意 strides 的张量。

### 18.2 block pointer 的 offsets 与旧版 offsets 有何区别？

旧版手动寻址中的 offsets 是 `(R, d)` 的逐元素偏移矩阵。新版 `make_block_ptr` 的 `offsets=(row_start, 0)` 则只描述**块左上角的二维坐标**，两者不是同一种对象。

```python
O_block_ptr = tl.make_block_ptr(
    O_ptr + batch * N_QUERIES * D,
    shape=(N_QUERIES, D), strides=(D, 1), offsets=(row_start, 0),
    block_shape=(ROW_TILE_SIZE, D), order=(1, 0),
)
```

参数逐项对应：

- base 先跳到当前 batch 的起始地址，一个 batch 跨过 Nq*d 个元素。
- shape 是该 batch 中整个 O 的形状，供边界判断使用。
- strides=(d,1) 表示跨一行移动 d 个元素，跨一列移动 1 个元素。
- offsets=(row_start,0) 表示块从指定行、第 0 列开始。
- block_shape=(R,d) 表示读取 R 行及全部 embedding 列。
- order=(1,0) 描述最后一维更连续的布局，不会转置矩阵。

block pointer 本身尚未包含读出的数值，`tl.load` 才产生 `(R,d)` 的 tensor。base 的偏移和 strides 均以元素为单位。

例如 batch=1、N=100、d=16、row_start=32，该块中的第 0 行第 5 列依然读取偏移 `(1*100+32)*16+5=2117` 的元素，数学地址与旧版本一致，只是不再显式生成完整地址矩阵。

block pointer 的 load 不同时传 `mask/other`，而使用 `boundary_check/padding_option`；store 同样使用 boundary_check。概率计算中的 causal/valid mask 仍需保留，因为内存边界检查不能替代数学遮罩。

## 19. flash_bwd_preprocess_kernel 逐段详解

这个 kernel 对应 Algorithm 2 的第 2 行，独立计算算法 D。

### 19.1 参数与 grid

输入是 O/dO 地址，输出是 Delta 地址。`N_QUERIES`、embedding `D`、`ROW_TILE_SIZE` 为 `tl.constexpr`，编译器据此确定加载和归约的形状。

包装函数的 grid 为 `(ceil(Nq/Bq), B)`，当前使用反向 query tile 大小作为 ROW_TILE_SIZE。每个 program 处理一个 batch 的一组 query 行。

### 19.2 构造 O、dO 和 Delta 的 block pointer

```python
batch = tl.program_id(1)
row_start = tl.program_id(0) * ROW_TILE_SIZE
O_block_ptr = tl.make_block_ptr(
    O_ptr + batch * N_QUERIES * D,
    shape=(N_QUERIES, D), strides=(D, 1), offsets=(row_start, 0),
    block_shape=(ROW_TILE_SIZE, D), order=(1, 0),
)
```

若行块大小为 32，program 2 的 row_start=64，加载 query 行 64–95。DO_block_ptr 的定义相同，只把 base 换为 DO_ptr。

Delta 每行只有一个值，所以其 block pointer 是一维：

```python
Delta_block_ptr = tl.make_block_ptr(
    Delta_ptr + batch * N_QUERIES,
    shape=(N_QUERIES,), strides=(1,), offsets=(row_start,),
    block_shape=(ROW_TILE_SIZE,), order=(0,),
)
```

一维 tuple 末尾的逗号不可省略，例如 `(row_start,)`。Delta 的 batch 起点不乘 embedding D，因为它的形状是 `(B,Nq)`。

### 19.3 boundary_check、补零与 float32 转换

```python
o = tl.load(O_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
```

`boundary_check=(0,)` 检查块的第 0 维，也就是 query 行。最后一块中超出 Nq 的行返回零，不访问无效地址，相当于旧版的行 mask 和 other=0。

embedding 不切块，并且 forward 要求 embedding 是至少 16 的 2 的幂，因此这里的 dims 正好覆盖全部 embedding，不需要额外列 mask。

注意先转换成 float32，再相乘与求和。若先在 BF16 中求乘积再转换，已经产生的低精度乘积舍入无法被撤销。

### 19.4 按行归约并写回

```python
delta = tl.sum(o * do, axis=1)
tl.store(Delta_block_ptr, delta, boundary_check=(0,))
```

`axis=1` 是 embedding 维，输出从 `(R, d)` 变为 `(R,)`，正好匹配 Delta_block_ptr 的一维块形状。store 通过 boundary_check 禁止写入越界 query 行。

所有有效 query 行都由某个 program 写入；尾块无效行禁止 store。

## 20. _flash_bwd_prob_and_ds：两个主 kernel 共用的局部计算

这是 `@triton.jit` 辅助函数，由主 kernel 在 GPU 计算中调用。它不是通过独立 grid 启动的新 kernel，也不会为了返回 P/dS 就将它们写入显存。

它的输入已经是局部 tensor：q/do 为 `(Bq, d)`，k/v 为 `(Bk, d)`，lse/delta 为 `(Bq,)`。

### 20.1 重算 scores

```python
scores = tl.dot(q, tl.trans(k), input_precision="tf32x3") * scale
```

`tl.trans(k)` 从 `(Bk, d)` 变为 `(d, Bk)`，dot 输出 `(Bq, Bk)`。这与前向计算 score 的方式一致。

### 20.2 causal mask 和有效位置 mask 分别解决什么？

```python
if is_causal:
    scores = tl.where(query_rows[:, None] >= key_rows[None, :], scores, scores - 1.0e6)
valid = (query_rows[:, None] < N_QUERIES) & (key_rows[None, :] < N_KEYS)
scores = tl.where(valid, scores, -float("inf"))
```

causal mask 屏蔽真实序列中的未来连接；valid mask 屏蔽 tile 补齐产生的不存在的 query/key。它们的用途不同。

对于有效 query 行和无效 key，score 必须为负无穷，才能让概率为零。否则即使越界 V 已补零，错误概率仍会影响梯度。

对于无效 query 行，主 kernel 加载的 lse/delta 为 0。此处 score=-∞，所以 P=exp(-∞-0)=0，后续梯度贡献也为零，不会污染有效 key 行的归约。

### 20.3 重算 P、dP、dS

```python
p = tl.exp(scores - lse[:, None])
dp = tl.dot(do, tl.trans(v), input_precision="tf32x3")
ds = p * (dp - delta[:, None])
return p, ds
```

lse 是每一行的全局归一化信息。两个 `[:, None]` 都将 `(Bq,)` 变为 `(Bq, 1)`，沿 key 维广播。

dp 的形状同样为 `(Bq, Bk)`。ds 使用逐元素乘法，既没有新的归约，也没有再次乘 scale。scale 会在最后求 dQ/dK 时使用。

在 dK/dV kernel 中 P 和 dS 都被使用；在 dQ kernel 中只用 dS。不过 dS 本身依赖 P，所以 dQ 路径依然必须重算概率。

## 21. flash_bwd_dkdv_kernel 逐段详解

### 21.1 固定 key 块

```python
batch = tl.program_id(1)
key_start = tl.program_id(0) * K_TILE_SIZE
key_rows = key_start + tl.arange(0, K_TILE_SIZE)
K_block_ptr = tl.make_block_ptr(
    K_ptr + batch * N_KEYS * D,
    shape=(N_KEYS, D), strides=(D, 1), offsets=(key_start, 0),
    block_shape=(K_TILE_SIZE, D), order=(1, 0),
)
```

这里的序列长度是 N_KEYS，不能误用 N_QUERIES：当 query/key 长度不同，batch 起始偏移也不同。

V、DK、DV 的 block pointer 使用相同的 shape、strides、offsets、block_shape，只是 base 指向不同 tensor。它们都固定在当前 key 块，不在 query 循环中推进。

K/V 用 `tl.load(..., boundary_check=(0,), padding_option="zero")` 读取，加载出的 k/v 在整个 query 循环中重复使用。key_rows 仍保留，但只用于辅助函数的 causal/valid mask，不再用于构造逐元素地址。

### 21.2 初始化累加器

```python
dk = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
dv = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
```

这是 program 内的局部零累加器。显存中的 dK/dV 使用 `torch.empty_like` 分配，暂时不初始化，等每个 program 完整累加后一次写入。

### 21.3 causal query_start 为什么向下对齐？

```python
query_start = 0
if is_causal:
    query_start = (key_start // Q_TILE_SIZE) * Q_TILE_SIZE
```

当前 key 块最早的 key 为 key_start。query 索引小于它的整块 query 都不可能关注当前块，可以跳过。

但要从**包含 key_start 的 query 块**开始，而不是向上对齐，否则可能漏掉有效 query。

例如 key_start=64、Bq=32，起点为 64，跳过 query 0–63。如果 key_start=32、Bq=64，起点为 0：块 0–63 的前半部分无效，但后半部分已有有效连接，需要靠局部 mask 判断，不能把整个块丢掉。

当前有限 -1e6 遮罩和跳块策略采用正常 score 范围下“被屏蔽概率数值上为零”的约定，具体限制见第 11 节。

### 21.4 每轮 query load

Q/DO/L/Delta 的 block pointer 在循环前创建，初始 offsets 使用已经由 causal 逻辑确定的 query_start。Q/DO 是二维块，L/Delta 是一维块。不能不论 causal 起点多少都从 0 加载，否则读取的 query 与 mask 的 query_rows 不一致。

每轮先加载指针当前指向的块：

```python
q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
lse = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
delta = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")
```

Q/dO 越界行补零；L/Delta 越界行也补零。辅助函数再把无效行的 P/dS 清零。

循环变量 start 用于构造全局 query_rows，以生成 mask。读取位置则由 block pointer 保存。每轮计算结束后，四个指针必须一起移动：

```python
Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
DO_block_ptr = DO_block_ptr.advance((Q_TILE_SIZE, 0))
L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
Delta_block_ptr = Delta_block_ptr.advance((Q_TILE_SIZE,))
```

二维 Q/DO 沿 query 维移动 Bq、embedding 维不动；一维 L/Delta 只移动 Bq。`.advance()` 返回新指针，所以需要赋值回变量。只推进 Q 而遗漏 L/Delta，会用下一块 query 配上上一块的归一化信息。

### 21.5 两次 dot 的转置、dtype 和 acc

```python
dv = tl.dot(tl.trans(p.to(do.dtype)), do, acc=dv, input_precision="tf32x3")
dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk, input_precision="tf32x3")
```

第一行的形状是 `(Bk, Bq) × (Bq, d)`，第二行完全相同，输出均为 `(Bk, d)`。

P/dS 在 softmax 相关计算中为 float32；做 BF16 dot 前转换成 do/q 的 dtype。acc 参数传入旧的 float32 累加器，dot 的结果直接与它累加。

转换的是本轮 dot 的输入，累加器没有每轮转回 BF16。最终累积所有 query 块后，dk/dv 才具备完整梯度。

### 21.6 最终 store

```python
tl.store(DK_block_ptr, (dk * scale).to(DK_block_ptr.type.element_ty), boundary_check=(0,))
tl.store(DV_block_ptr, dv.to(DV_block_ptr.type.element_ty), boundary_check=(0,))
```

dk 在存储前统一乘 scale；dv 不乘。`block_ptr.type.element_ty` 表示块中元素的类型，转换保证按对应梯度 dtype 存储。输出 block pointer 始终固定在当前 key_start，boundary_check 屏蔽无效 key 行。

若某个 causal key 块没有任何有效 query，循环可以不执行，零累加器仍会写回，得到零梯度，而不是留下未初始化数据。

## 22. flash_bwd_dq_kernel 逐段详解

### 22.1 固定 query 块

query_start 由 program 第 0 维确定。Q/DO/DQ 的 block pointer 都指向 `(query_start,0)`；L/Delta 指向 `(query_start,)`。Q/dO/L/Delta 全在循环前加载，DQ 只用于最终 store。这五个指针在 key 循环内都不推进。

初始化 dq 的形状为 `(Bq, d)`，dtype 为 float32。它对当前 query 块负责，其他 query-program 的写入范围与它不重叠。

### 22.2 causal key_end 为什么这样算？

```python
key_end = N_KEYS
if is_causal:
    key_end = tl.minimum(N_KEYS, query_start + Q_TILE_SIZE)
```

Python 风格的 range 上界不包含 key_end。当前 query 块最大的索引是 `query_start+Bq-1`，所以最多需要覆盖该索引及之前的 key。

例如 query_start=64、Bq=32，最大 query 为 95，key_end=96。若 Bk=64，循环仍会加载 key 0–63 和 64–127 两个块；后一块中索引 96–127 靠局部 causal mask 清零。

缩小循环上界只跳过整块无效计算，不能代替对角附近的逐元素遮罩。

### 22.3 key 循环、辅助函数和 dot

K/V 的 block pointer 在循环前从 offsets=(0,0) 开始。每轮从它们加载 k/v，另外用 start 构造全局 key_rows，供 mask 使用。辅助函数返回 P 和 dS，这里用 `_` 忽略 P 的返回值。

```python
dq = tl.dot(ds.to(k.dtype), k, acc=dq, input_precision="tf32x3")
```

这里**不转置 dS**：`(Bq, Bk) × (Bk, d) -> (Bq, d)`。沿 key 维归约，对固定 query 块累计贡献。

每轮计算结束后：

```python
K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
```

只推进 key 维，不移动 embedding 维。下一轮的 key_rows 也增加 Bk，与指针指向的新块一致。

循环完成后统一乘 scale，转换成 DQ 的元素 dtype，再用固定的 DQ_block_ptr 写回：

```python
tl.store(DQ_block_ptr, (dq * scale).to(DQ_block_ptr.type.element_ty), boundary_check=(0,))
```

代码与 dK/dV 互为不同方向的块归约。block pointer 负责内存边界，辅助函数里的 valid/causal mask 负责保证数学贡献正确。

## 23. FlashAttentionTriton：forward 为 backward 做了哪些准备？

### 23.1 前向和反向 tile 可以不同

当前类属性为：

```python
Q_TILE_SIZE = 16
K_TILE_SIZE = 16
BWD_Q_TILE_SIZE = 32
BWD_K_TILE_SIZE = 64
```

前向默认 16/16，反向默认 32/64。它们不需要相同，因为反向通过保存的 L 重算任意局部 P，不依赖前向如何切块。

benchmark 可以修改这些类属性进行调优。forward 会检查四个 tile 大小都是至少 16 的 2 的幂，但这个检查不保证任意大 tile 都能满足 GPU 的寄存器和 shared memory 限制。

### 23.2 输入检查为什么与反向相关？

当前包装类要求三维 Q/K/V，batch 和 embedding 一致，K/V 形状相同，dtype/device 一致。支持 FP16、BF16、FP32；embedding 为至少 16 的 2 的幂，序列长度为正。

这些限制保证 block pointer 中的 strides=(d,1)、tl.arange 和 dot 的输入形状可用。当前实现没有直接接收 `(B, H, N, d)`；若未来增加多头，需要同步修改形状和 grid 映射。

### 23.3 保存张量与参数

```python
ctx.save_for_backward(lse, q, k, v, o)
ctx.is_causal = is_causal
ctx.bwd_q_tile_size = FlashAttentionTriton.BWD_Q_TILE_SIZE
ctx.bwd_k_tile_size = FlashAttentionTriton.BWD_K_TILE_SIZE
```

保存的 q/k/v 是已经连续化的输入。lse 是前向实际算出的归一化信息，o 是返回的实际输出，两者用于重算概率和 D。

保存 tile 参数的作用是让这一轮 forward/backward 使用同一份反向配置。如果创建计算图后又修改类属性，旧计算图仍使用当时保存的值。

## 24. FlashAttentionTriton.backward 逐段详解

### 24.1 取回数据并处理上游梯度

```python
lse, q, k, v, o = ctx.saved_tensors
grad_o = grad_o.contiguous()
```

PyTorch 传入的 grad_o 可能来自转置或广播视图，不一定连续。连续化后，三个 kernel 才能使用预设的 block pointer strides。它已经连续时通常不需要复制；否则这次转换会产生拷贝，其开销属于实际 backward 路径。

### 24.2 分配中间结果和梯度

```python
delta = torch.empty_like(lse)
d_q, d_k, d_v = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
scale = 1.0 / math.sqrt(d)
```

lse 为 float32，因此 delta 也是 float32，形状 `(B, Nq)`。输入梯度的形状和 dtype 分别匹配 Q/K/V。

本次只额外分配一个逐行 Delta 向量和三个必要的梯度输出，没有分配 S/P/dP/dS 的完整矩阵。

### 24.3 三次 launch

使用 `with torch.cuda.device(q.device)` 选择输入所在设备，依次启动：

- preprocess：`(ceil(Nq/Bq), B)`，写 Delta。
- dK/dV：`(ceil(Nk/Bk), B)`，读 Delta，写 dK/dV。
- dQ：`(ceil(Nq/Bq), B)`，读 Delta，写 dQ。

Nq/Nk/d/tile/causal 是 kernel 的编译时参数；scale 是数学缩放值。每个 kernel 的启动参数顺序必须与其定义对应。

同一 stream 中的顺序保证预处理完成后主 kernel 才读取 Delta。包装函数不做主机同步，保持 CUDA 异步执行；测试或计时工具在需要读结果或统计耗时时负责等待。

### 24.4 为什么最后要切片梯度 tuple？

```python
return (d_q, d_k, d_v, None)[:len(ctx.needs_input_grad)]
```

显式调用 `.apply(q, k, v, is_causal)` 时，autograd 有四个输入位置，需要返回四个梯度位置。

调用 `.apply(q, k, v)` 时，forward 通过默认值取得 is_causal=False，但调用者实际只传了三个输入。切片使返回 tuple 长度对应实际输入数量。

`ctx.needs_input_grad` 的每个元素还表示对应输入是否需要梯度；当前代码只用它的长度，仍计算所有 dQ/dK/dV，没有据此跳过不需要的梯度。

这组 raw Triton kernel 实现当前的一阶反向计算，没有为这些 kernel 运算构建可再求导的 PyTorch 计算图；不能因为继承 autograd.Function 就认为二阶梯度也已经实现。

## 25. 用一组具体尺寸检查整个执行过程

假设 B=2、Nq=96、Nk=80、d=64，反向 Bq=32、Bk=64：

```text
preprocess grid: (3, 2)
dK/dV grid:      (2, 2)
dQ grid:         (3, 2)
```

preprocess 每个 batch 有三个行块，分别计算 query 0–31、32–63、64–95 的 D。

非 causal 时，每个 key-program 遍历三个 query 块。key-program 0 负责 key 0–63；key-program 1 负责 key 64–79，并在其 64 行局部 tile 中将 80–127 的无效行屏蔽。

每个 query-program 遍历两个 key 块，最终只写自己的 32 行 dQ。尽管 Nq 与 Nk 不相同，Q/dO 使用 Nq 的 batch 步长，K/V 使用 Nk 的 batch 步长，因此地址不会混淆。

causal 时，key-program 1 从 query 64 开始，只处理最后一个 query 块。query-program 0 的 key_end=32，但会加载第一个 64-key 块，由 mask 屏蔽超过当前 query 的 key；query-program 2 的 key_end=min(80,96)=80，遍历两个 key 块。

这一例子同时说明了三点：grid 的第 0 维取决于最终写入的梯度块；循环方向取决于还需要累加的另一维；tile 里的有效位置最终由全局索引 mask 决定。

对应测试可在 CUDA 机器运行：

```bash
uv run pytest -k test_flash_backward_triton_tiled
```

测试定义了三组形状，分别覆盖不同长度和 embedding；再与两种 dtype、两种 causal 模式做组合，共 12 项测试。它还使用非连续 dO，并在 non-causal 分支省略 is_causal 参数，以检查上游梯度转换和默认参数的梯度返回数量。

本次文档更新只核对代码与说明，没有在本地执行 CUDA kernel 或新增 GPU 测量结果。
