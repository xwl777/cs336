# FlashAttention-2 前向：任务要求、实现思路与代码详解

本文对应作业 `Problem (flash_forward)` 的 (a)、(b)、(c)，解释当前实现，不修改实现代码。

相关文件：

- [实现：flash_attention.py](cs336_systems/flash_attention.py)
- [测试适配接口：adapters.py](tests/adapters.py)
- [正确性测试：test_attention.py](tests/test_attention.py)

## 1. 这道题最终要实现什么？

输入是 Q、K、V，输出是普通 attention 的结果 O：

$$
S = \frac{QK^\top}{\sqrt d},\qquad
P = \operatorname{softmax}(S),\qquad
O = PV.
$$

FlashAttention 改变的是计算和存储方式。它把矩阵分块，在处理每个块时更新结果，避免将完整的 S 和 P 写入显存。算法在精确算术下等价于普通 attention；实际运行时，浮点计算顺序和精度可能造成小误差。

还需要计算每一行的 logsumexp：

$$
L_i = \log\sum_j \exp(S_{ij}).
$$

它用于以后重算 attention 概率：

$$
P_{ij} = \exp(S_{ij} - L_i).
$$

题目开头提到产生 O 和 L，但随后明确规定 `autograd.Function.forward` 返回 O，并把 L 保存到上下文。当前实现遵循这个接口，返回值不是 `(O, L)`。

### 1.1 (a)：纯 PyTorch 版本

要求定义一个 `torch.autograd.Function` 子类，用 PyTorch 操作实现 Algorithm 1：

- forward 参数是 `ctx, Q, K, V, is_causal=False`。
- 使用 query/key 分块和 online softmax，块大小至少为 16 × 16。
- 保存 L、Q、K、V、O，返回 O。
- 这一小问可以忽略 causal mask。
- 必须定义 backward，但本题允许暂时抛出 `NotImplementedError`。
- 在 `adapters.get_flashattention_autograd_function_pytorch` 中返回这个类。

这个版本主要用于理解算法和对照调试。Python 循环加上许多小 PyTorch 操作，通常比普通 attention 慢。

### 1.2 (b)：Triton 版本

要求把同样的前向算法写成 Triton kernel，并用另一个 `autograd.Function` 包装：

- launch grid 为 `(Tq, batch_size)`，其中 Tq 是 query 块数。
- 每个 program 负责一个 batch 中的一个 query 块。
- kernel 内只有一个循环，用于遍历 K/V 块。
- 用 block pointer 访问块，用 `tl.dot` 做矩阵乘法。
- 循环末尾推进 K/V block pointer。
- 输出累加器、运行最大值 m、运行分母 l 使用 float32。
- 在乘 V 前，把未归一化概率转换为 V 的 dtype。
- 写 O 前转换为输出 tensor 的 dtype。
- 在 `adapters.get_flashattention_autograd_function_triton` 中返回包装类。

题目正文有一处把接口简称为 `get_flash_autograd_function_triton`；当前仓库实际使用的名字是 `get_flashattention_autograd_function_triton`，以测试导入的名字为准。

### 1.3 (c)：causal mask

增加可选布尔参数 `is_causal=False`。当它为 True 时，第 q 行只允许关注索引 k ≤ q 的 key。

Triton kernel 中该参数必须标注为 `tl.constexpr`，通过 query/key 的全局索引比较生成 mask，对禁止的位置给 score 加上 -1e6，并通过 `ctx.is_causal` 保存标志。

题目保证测试维度是至少 16 的 2 的幂。当前块大小为 16，因此这些维度可以被块大小整除。

## 2. 张量形状与符号

本文用 B 表示 batch size，Nq/Nk 表示 query/key 数量，d 表示 embedding dimension。没有多头维度。

```text
Q: (B, Nq, d)
K: (B, Nk, d)
V: (B, Nk, d)
O: (B, Nq, d)
L: (B, Nq)
```

当前实现中，V 的最后一维也取 d。Bq、Bk 分别表示 query 和 key 的块大小，当前均为 16。

对一个 batch 的一个 query 块，内部形状是：

```text
Q_tile:        (Bq, d)
K_tile:        (Bk, d)
V_tile:        (Bk, d)
S_tile:        (Bq, Bk)
P_tilde:       (Bq, Bk)
输出累加器 A: (Bq, d)
m、l、alpha:   (Bq,)
```

PyTorch 版本一次处理各个 batch，因此这些形状前面还带有 B；Triton 的单个 program 已选定一个 batch，所以块内部没有 batch 维度。

## 3. 为什么要用 online softmax？

普通 softmax 需要整行分母：

$$
P_{ij}=\frac{\exp(S_{ij})}{\sum_t\exp(S_{it})}.
$$

只看一个 key 块时，还不知道后续块的分母，因此不能把每个块独立做 softmax 后直接相加。

直接在线累加 `exp(S)` 也有数值问题。例如 S = 1000 时，float32 的指数会溢出。先算指数和、最后再除以最大值的指数，不能挽救已经发生的溢出。

稳定计算会减去行最大值 m：

$$
P_{ij}=\frac{\exp(S_{ij}-m_i)}{\sum_t\exp(S_{it}-m_i)}.
$$

减去相同常数不改变 softmax。但分块过程中 m 可能增加，所以旧的累计结果必须随着 m 更新而调整尺度。

### 3.1 三个运行状态的准确含义

对一行 query，设已处理 key 的集合是 J。为了避免把“未归一化输出”和最终输出混淆，本文将代码中的 `o_tile` / `acc` 记为 A：

$$
m = \max_{t\in J} S_t,
$$

$$
l = \sum_{t\in J}\exp(S_t-m),
$$

$$
A = \sum_{t\in J}\exp(S_t-m)V_t.
$$

初始化为 m = -∞、l = 0、A = 0。处理完全部 key 后，最终输出是 A/l。

### 3.2 加入一个新 key 块时如何更新？

先计算新最大值：

$$
m_{new}=\max(m_{old},\operatorname{rowmax}(S_{tile})).
$$

新块按新最大值计算指数：

$$
\widetilde P=\exp(S_{tile}-m_{new}).
$$

旧指数的尺度转换系数为：

$$
\alpha=\exp(m_{old}-m_{new}).
$$

因为 `exp(S - m_new) = exp(S - m_old) × alpha`，旧分母和旧输出都乘 alpha：

$$
l_{new}=\alpha l_{old}+\operatorname{rowsum}(\widetilde P),
$$

$$
A_{new}=\alpha A_{old}+\widetilde P V_{tile}.
$$

这里 alpha 是逐行缩放；算法中的 `diag(alpha) A` 就是代码中的 `alpha[:, None] * A`，无需构造对角矩阵。

全部块处理完后：

$$
O=A/l,\qquad L=m+\log l.
$$

### 3.3 一个单行数值例子

第一块只有 score = 2，对应 value = 10。处理后 m = 2、l = 1、A = 10。

第二块只有 score = 3，对应 value = 20。新最大值为 3，因此 alpha = exp(2 - 3) = exp(-1)：

$$
l=e^{-1}+1,\qquad A=10e^{-1}+20.
$$

最终 O = A/l，约为 17.31，与对 `[2, 3]` 做普通 softmax 后加权求和相同。乘 alpha 保留了第一块与第二块的正确相对权重。

## 4. 文件的整体结构

`flash_attention.py` 包含四个主要部分：

1. `_flash_backward_pytorch`：供 PyTorch 类使用的 compiled PyTorch 反向计算。
2. `FlashAttentionPytorch`：纯 PyTorch 分块前向和对应 autograd 接口。
3. `flash_fwd_kernel`：Triton 分块前向 kernel。
4. `FlashAttentionTriton`：分配输出、启动 kernel、保存数据的包装类；现已接入三个分块反向 kernel，详见 [反向说明](FLASH_ATTENTION_TRITON_BACKWARD_GUIDE.md)。

Triton 使用可选导入：未安装 Triton 时，PyTorch 版本仍可使用；调用 Triton 包装类时会给出错误。这个模块本身没有计时或训练入口。

## 5. PyTorch forward 逐段解释

### 5.1 为什么继承 autograd.Function？

自定义 `torch.autograd.Function` 让你明确指定 forward 如何计算，以及 backward 如何从输出梯度计算输入梯度。

调用入口是 `FlashAttentionPytorch.apply(...)`，而不是手动创建实例。PyTorch 自动提供 ctx；ctx 用来在 forward 和 backward 之间传递数据。

自定义 Function 的 forward 内部默认不会记录普通 autograd 计算图，因此 backward 需要由这个类显式提供。

### 5.2 缩放与 tile 大小

`d = q.shape[-1]` 取得 embedding 维度，`scale = 1.0 / math.sqrt(d)` 对应 attention 的缩放系数。

`q_tile_size = k_tile_size = 16` 是当前实现选择的块大小，不代表最优性能配置。

### 5.3 外层 query 循环

`q_start` 每次增加 16，切片取得 `q_tile`，形状是 `(B, Bq, d)`。`...` 表示保留前面的维度。

为这个 query 块初始化：

- `o_tile`：未归一化输出累加器，形状 `(B, Bq, d)`。
- `m`：每行最大值，形状 `(B, Bq)`，float32。
- `l`：每行分母，形状 `(B, Bq)`，float32。
- `query_positions`：这个块各行在完整序列中的索引。

注意当前 `o_tile` 初始使用 `zeros_like(q_tile)`，dtype 跟随 q；后续运算可能提升 dtype。Triton 版本则显式从一开始使用 float32 累加器。

### 5.4 内层 key 循环与 einsum

`k_start` 每次增加 16，同时取对应的 K 块和 V 块。

代码中的 einsum：

```python
torch.einsum("... q d, ... k d -> ... q k", q_tile, k_tile)
```

表示对 d 维乘积求和，保留 query 和 key 维，等价于 `q_tile @ k_tile.transpose(-1, -2)`。结果为 `(B, Bq, Bk)`。

另一处 einsum：

```python
torch.einsum("... q k, ... k d -> ... q d", p_tilde, v_tile)
```

表示对 key 维求和，得到当前块对输出的贡献，形状为 `(B, Bq, d)`。

### 5.5 online softmax 与广播

`torch.max(s, dim=-1).values` 沿 key 维取最大值，得到 `(B, Bq)`；`torch.maximum` 再与旧 m 逐元素比较。

`m_next[..., None]` 在最后增加一个长度为 1 的维度：

```text
s:                     (B, Bq, Bk)
m_next[..., None]:     (B, Bq, 1)
s - m_next[..., None]: (B, Bq, Bk)
```

因此每个 query 行的所有 key score 都减去同一个最大值。

`alpha[..., None] * o_tile` 同理：`(B, Bq, 1)` 广播到 `(B, Bq, d)`，对输出每一行整体缩放。

更新 l 时沿最后一个 key 维求和。更新输出时，先用 `p_tilde.to(v.dtype)` 转换类型，再与 V 块相乘。

### 5.6 最终归一化、拼接和保存

处理完所有 K/V 块后，`o_tile / l[..., None]` 得到当前 query 块的最终输出。`m + torch.log(l)` 得到该块的 L。

各块分别加入列表，最后：

- `torch.cat(outputs, dim=-2)` 沿 query 维拼接 O。
- `torch.cat(lse_tiles, dim=-1)` 沿 query 维拼接 L。

`ctx.save_for_backward(lse, q, k, v, o)` 保存反向所需张量；保存张量不等于把每个张量再复制一份。`ctx.is_causal` 保存普通布尔值，最后返回 O。

## 6. Triton 包装类逐段解释

### 6.1 检查输入和分配空间

`FlashAttentionTriton.forward` 检查 Triton 是否可用，以及 Q/K/V 是否都在 CUDA 上。当前包装类按三维 `(B, N, d)` 输入解包。

`.contiguous()` 保证连续布局；输入已经连续时通常不会复制，不连续时可能创建副本。

`torch.empty_like(q)` 为 O 分配空间；`torch.empty((B, Nq), dtype=torch.float32, ...)` 为 L 分配空间。`empty` 不初始化元素，kernel 必须写完所有有效输出。

### 6.2 二维 grid

实际设置为：

```python
grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
```

`triton.cdiv` 是向上取整除法。例如 B = 8、Nq = 128、Bq = 16，grid 为 `(8, 8)`，共有 64 个 program。

第一维选择 query 块，第二维选择 batch。program `(2, 3)` 负责 batch 3 中索引 32 到 47 的 query 行，计算这些行对应的 O 和 L。

这里一个 program 是 Triton 的调度单位，不能理解为“一个线程处理一个完整 tile”。program 内有多个 GPU 线程协作，具体布局由编译器安排。

### 6.3 kernel 参数为什么那么多？

除了张量地址，还需要传入 strides，以便 kernel 计算元素地址。比如 Q 的三个 stride 分别表示沿 batch、query、embedding 维移动一个元素时，地址跨过多少个元素。

连续 `(B, Nq, d)` 张量通常有 strides `(Nq*d, d, 1)`。stride 的单位是元素，不是字节。

`D`、`Q_TILE_SIZE`、`K_TILE_SIZE`、`is_causal` 标注为 `tl.constexpr`，表示编译时确定。编译器可以据此确定块形状，并为 causal/non-causal 情况生成不同版本。

## 7. Triton kernel 逐段解释

### 7.1 program 编号

```python
query_tile_index = tl.program_id(0)
batch_index = tl.program_id(1)
```

它们分别读取二维 grid 的第一维和第二维索引。

PyTorch 版本在 Python 外层循环里遍历 query 块；Triton 把这些 query 块分配给不同 program，所以 kernel 内不再需要 query 循环，只保留 key 循环。

### 7.2 block pointer 的含义

以 Q 为例，`Q_ptr + batch_index * stride_qb` 先找到对应 batch 的起始地址。之后建立二维 block pointer：

- `shape=(N_QUERIES, D)`：该 batch 中完整 Q 的形状。
- `strides=(stride_qq, stride_qd)`：query 和 embedding 维的跨度。
- `offsets=(query_tile_index * Q_TILE_SIZE, 0)`：当前块的起始坐标。
- `block_shape=(Q_TILE_SIZE, D)`：一次读取的块形状。
- `order=(1, 0)`：指定维度顺序；这里最后一维连续，便于编译器优化访问。

O 指针指向同样的 query 行。L 是一维，每个 query 行只写一个数。K/V 指针从该 batch 的第一个 key 块开始，循环中不断推进。

block pointer 是内存访问描述，不是已经加载到芯片上的 tensor；调用 `tl.load` 才实际读取块。

### 7.3 加载 Q 和初始化状态

Q 块只加载一次。`acc` 形状为 `(Bq, d)`，m/l 形状为 `(Bq,)`，都使用 float32。`acc` 对应前文的未归一化输出 A。

`boundary_check` 和 `padding_option="zero"` 可以防止部分越界读取。当前代码还根据全局 key 索引将无效 key 的 score 设为负无穷；仅把越界 K/V 补零会给 softmax 分母加入错误贡献。

### 7.4 唯一的循环：遍历 key 块

每一轮加载一个 K 块和一个 V 块，并计算：

```python
s = tl.dot(q, tl.trans(k)) * scale
```

形状变化是 `(Bq, d) × (d, Bk) -> (Bq, Bk)`。这是 Triton kernel 内的矩阵乘法接口。

随后用 `tl.max(s, axis=1)` 得到行最大值，计算 m_next、p_tilde、alpha 和 l。`axis=1` 是 key 维，因为 kernel 内的 s 是二维。

输出更新为：

```python
acc = tl.dot(p_tilde.to(V_block_ptr.type.element_ty), v, acc=acc * alpha[:, None], input_precision="tf32x3")
```

`V_block_ptr.type.element_ty` 是 V 指针指向的元素类型。转换 p_tilde 的类型是为了满足后续 dot 的输入类型要求，acc 仍用于 float32 累加。

当前代码采用 `tl.dot(..., acc=...)`，传入逐行缩放后的累加器；FP32 dot 显式选择 `tf32x3` 精度策略。

### 7.5 推进 K/V 指针

```python
K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
key_offsets += K_TILE_SIZE
```

移动的是二维坐标中的 key 维。embedding 维不切块，因此第二个增量为 0。Q 和输出位置不变，因为这个 program 始终负责同一组 query 行。

### 7.6 写回 O 和 L

循环结束后计算 `o = acc / l[:, None]`、`lse = m + tl.log(l)`，用 `tl.store` 写入对应位置。O 在存储前转换成输出元素 dtype。

不同 program 对应不同 batch 或不同 query 行，写入区域互不重叠，所以写 O/L 不需要原子加法或跨 program 求和。

## 8. causal mask 怎样工作？

关键是使用完整序列中的索引，而不只是块内索引：

```python
query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
key_offsets = tl.arange(0, K_TILE_SIZE)
```

构造 mask 时：

```text
query_offsets[:, None]: (Bq, 1)
key_offsets[None, :]:   (1, Bk)
比较结果:               (Bq, Bk)
```

条件 `query_offsets[:, None] >= key_offsets[None, :]` 在 k ≤ q 时为 True。

例如 query 全局索引为 2，只有 key 索引 0、1、2 允许参与；key 索引 3 及之后的位置被屏蔽。当前实现包含对角线，因此允许关注自己。

Triton 代码在禁止位置使用 `s + -1.0e6`，符合本题要求。PyTorch 代码使用 `torch.where(mask, s, -1e6)`，直接替换被屏蔽的 score。两种写法在被屏蔽处并非代数上相同，但对正常幅度的本题输入，指数权重都近似为零。

这个有限常数依赖输入范围，不能当作任意 score 范围下严格的负无穷替代。当前实现也没有跳过未来的 K/V 块，因此启用 causal mask 不代表计算量自动减半。

## 9. 当前 backward 在做什么？

前向题本身只要求前向；当前文件的 PyTorch 类使用 compiled PyTorch 反向，下面解释它的公式。Triton 类现使用分块 Triton 反向，详见 [Algorithm 2 说明](FLASH_ATTENTION_TRITON_BACKWARD_GUIDE.md)。

先从 ctx 取回 L、Q、K、V、O，再重新计算 S 和 P：

$$
S=QK^\top/\sqrt d,\qquad P=\exp(S-L).
$$

给定上游梯度 dO，接下来计算：

$$
dV=P^\top dO,\qquad dP=dO V^\top,
$$

$$
D_i=\sum_t O_{it}dO_{it},\qquad
dS_{ij}=P_{ij}(dP_{ij}-D_i),
$$

$$
dQ=dS K/\sqrt d,\qquad dK=dS^\top Q/\sqrt d.
$$

代码中的 `d_scalar` 对应这里的 D，和 embedding 维度 d 不是同一个含义。

`backward` 返回 `(d_q, d_k, d_v, None)`，对应显式传入的四个 forward 参数。布尔值 is_causal 不需要梯度，因此最后是 None；ctx 不占返回梯度的位置。测试显式传入了第四个参数。

需要区分两个事实：前向没有保存完整 P；但 PyTorch 参考 backward 会重新构造完整 S、P、dP、dS，仍有 O(B*Nq*Nk) 中间张量成本。新接入的 Triton backward 只重算局部块，不再创建这些完整矩阵。

## 10. 内存节省在哪里？

当 Nq = Nk = N 时，普通 attention 的 S/P 每个有 B*N² 个元素。float32 下，每个矩阵占 `4*B*N²` 字节。

例如 B = 8、N = 16384，一个完整矩阵就占 8 GiB。若前向和反向同时保留多个这种矩阵，显存很容易不足。

分块前向只需要局部 Bq × Bk 的 score，以及 Bq × d 的输出累加器，并将最终 O/L 写回显存。为 backward 保存的数据是 Q/K/V/O/L，总体随序列长度线性增长（d 固定时），不再保存 N × N 的 P。

这里的复杂度描述不代表精确的 GPU 峰值显存：峰值还取决于 dtype、输入副本、PyTorch 临时量、编译器布局和运行时分配。当前参考反向的平方级临时内存也需要单独考虑。

FlashAttention 对一般稠密 attention 的算术量仍是平方级；主要收益来自减少显存读写和中间存储，而不是省略 attention 的数学计算。

## 11. adapters 和测试分别检查什么？

`tests/adapters.py` 返回类对象，测试随后调用该类的 `.apply`。这里返回的是类，而不是类的实例或一次 attention 的结果。

当前前向测试创建 B = 4、Nq = Nk = 128、d = 64 的输入，检查：

1. 输出的 `grad_fn.saved_tensors` 存在。
2. 保存的张量里恰好有一个形状 `(B, Nq)` 的张量，作为 L。
3. O 与普通 attention 参考实现接近。
4. L 与 `torch.logsumexp(S, dim=-1)` 接近。

比较容差为 `rtol=1e-2, atol=1e-2`。Triton 前向测试分别覆盖 `is_causal=False` 和 True；当前 PyTorch 前向测试只覆盖非 causal 情况。因此一个测试通过不能证明所有 dtype、形状和输入范围都正确。

在项目目录运行：

```bash
cd /Users/lwx/cs336/assignment2-systems
uv run pytest -k test_flash_forward_pass_pytorch
uv run pytest -k test_flash_forward_pass_triton
```

额外的参考反向测试：

```bash
uv run pytest -k test_flash_backward_pytorch
uv run pytest -k test_flash_backward_triton
```

截至本文整理时，此前执行的 PyTorch 前向和反向测试通过；Triton 测试因当前机器无 CUDA 而跳过。跳过不代表 kernel 已编译运行成功，仍需在支持 Triton 的 CUDA 环境中验证。

## 12. 建议按什么顺序读代码？

1. 先理解普通 attention 的 S、P、O 和每个张量的形状。
2. 读 PyTorch forward 的两个循环，明确每次取了哪一块 Q/K/V。
3. 对照 m、l、A 的定义，理解 alpha 为什么同时缩放旧分母和旧输出。
4. 读 Triton 包装类的 grid，明确一个 program 的责任范围。
5. 对照 PyTorch 操作读 kernel，检查每个 tl.dot 的输入输出形状。
6. 最后阅读 causal mask、ctx 保存数据和参考 backward。

读到每一次循环结束时，可以检查三个不变量：m 是已处理 key 的行最大值；l 是相对 m 的指数和；acc 是同一尺度下的 value 加权和。这三个定义成立，最终的 acc/l 就能得到正确输出。
