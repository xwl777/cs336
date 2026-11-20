# Overlap DDP：先理解训练，再读这份代码

对照实现：[overlap_ddp.py](../cs336_systems/overlap_ddp.py)。
调用方：[benchmark_naive_ddp.py](../scripts/benchmark_naive_ddp.py) 的 `train_step`。

先记住一句话：**DDP 让每张卡各算一部分数据的梯度，再把梯度平均，使大家用同一个梯度更新模型。Overlap 想进一步做到：一个参数的梯度算好了，就先通信，不必等其他参数都算完。**

当前实现把两个阶段分开：hook 只发起通信、保存任务凭证；整个 backward 结束后，finish 才等待并求平均。这样剩余反向计算可以与已经发起的通信重叠，实际加速程度仍需 GPU 测量。

## 1. DDP 到底要解决什么？

假设有两张 GPU，每张上面都放着同一个模型。这两份模型的参数起初相同，但是读的数据不同：

- rank 0（第 0 个进程）读前 10 条数据。
- rank 1（第 1 个进程）读后 10 条数据。

`rank` 就是进程编号；`world_size` 就是参加训练的进程数，这里是 2。通常一个进程负责一张 GPU。

只看模型中的一个参数 `w`。假设两边算出的本地梯度分别是 2 和 6。

如果直接更新，两边会分别做 `w -= lr * 2` 和 `w -= lr * 6`，模型就不一样了。

所以先交换并合并梯度：

```text
                          rank 0       rank 1
backward 后的 w.grad         2            6
all_reduce(SUM) 后           8            8
除以 world_size 后           4            4
optimizer 使用的梯度         4            4
```

**all-reduce 既求和，又把结果放回所有参与进程。** 不是只把结果交给 rank 0。

当各 rank 的样本数相同、loss 是本地样本的平均值时，两边梯度取平均，就对应把这 20 条数据一起算平均 loss 的梯度。样本数不相等时，简单除以进程数不等于按样本平均。

还有一个容易混淆的地方：`w` 是模型参数，`w.grad` 是它的梯度。这里 all-reduce 改的是 **`w.grad`**，更新 **`w`** 是稍后 `optimizer.step()` 的工作。

## 2. 为什么能提前开始通信？

想象一个简单串行网络：

```text
forward：   输入 → A层 → B层 → C层 → loss
backward：  loss → C层 → B层 → A层
```

反向传播沿计算图往回走。因此，在这个例子里，C 层参数的梯度可以先就绪，A 层参数的梯度稍后才就绪。

**不是等整个 `loss.backward()` 返回，所有参数才突然一起有梯度。** backward 进行到一半时，已经有一部分参数的 `.grad` 可用了。

这给了我们一个机会：C 的梯度一出来，就把它送去 all-reduce，同时继续计算 B、A 的梯度。

```text
等全部算完再通信：
计算：  [ C反向 ][ B反向 ][ A反向 ]
通信：                           [ C通信 ][ B通信 ][ A通信 ]

理想重叠（示意，不代表实际耗时或通信彼此并发）：
计算：  [ C反向 ][ B反向 ][ A反向 ]
通信：           [ C通信 ][ B通信 ][ A通信 ]
```

目标是减少整个训练 step 的时间。最后通常仍有没被计算遮住的通信尾巴，不能默认通信开销全部消失。

复杂网络、分支和共享参数的就绪顺序由计算图决定，不能机械理解为“严格按层编号倒序”。这份简化实现要求各 rank 的 hook 发起兼容且相同顺序的 collective，否则可能配错或卡住；它没有实现完整 DDP 的调度机制。

## 3. 一次训练的完整顺序

先看整体，暂时不用钻进 hook：

```python
# 前提：调用方已经初始化进程组、分配设备、切分数据
model = OverlapDDP(base_model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(local_x), local_y)
loss.backward()                          # 内部自动触发各参数的 hook
model.finish_gradient_synchronization() # 在更新前建立通信完成的依赖
optimizer.step()                        # 用平均后的梯度更新参数
```

`OverlapDDP` 不会替你切分数据，也不会替你调用 optimizer。它负责初始状态对齐，以及梯度同步。

## 4. 构造函数：让所有人从同一个起点出发

```python
self.module = module
self.world_size = dist.get_world_size()
self.handles: list[tuple[torch.Tensor, dist.Work]] = []
self._sync_enabled = True
```

- `module`：被包装的原模型。
- `world_size`：稍后梯度求平均时的除数。
- `handles`：保存 `(parameter, handle)`，把参数和对应通信任务配对，收尾时才知道该给哪个梯度做除法。handle 是 `dist.Work` 对象，不是梯度或求和结果。
- `_sync_enabled`：当前 backward 是否允许通信。

构造函数先检查进程组已经初始化。进程组可以理解为“哪些进程参加这次通信”的约定。

接着用 `dist.broadcast(..., src=0)` 把 rank 0 的参数和 buffer 复制到其他 rank。**先保证模型相同，再保证每一步的梯度相同，才能保持模型一致。**

冻结参数虽然不训练，也影响模型计算，所以仍然要广播。buffer 是模型登记的非参数状态；当前代码只在初始化时广播它们，没有在每次 forward 前同步。

`torch.no_grad()` 表示这次初始复制不需要被 autograd 记录。

## 5. 注册 hook：让 PyTorch 在梯度就绪时叫我们

```python
for parameter in self.module.parameters():
    if not parameter.requires_grad:
        continue
    parameter.register_post_accumulate_grad_hook(self._allreduce_gradient)
```

这里不是马上调用 `_allreduce_gradient`，而是给参数安装回调：**以后 backward 更新完这个参数的 `.grad`，就调用它。**

`post_accumulate` 的意思是“累加到 `.grad` 之后”。回调收到的是参数 `parameter`，通过 `parameter.grad` 才拿到梯度。

[PyTorch 的 hook 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html)确认：此时该叶子张量的 `.grad` 已更新，回调可以访问、原地修改它。

这里的“累加完成”是这一次 backward 对这个参数的累加完成，**不代表整个 backward 结束，更不代表所有 micro-batch 都跑完了**。

冻结参数不需要 hook。共享权重如果是同一个 `Parameter` 对象，`module.parameters()` 默认会去重；多个使用位置的贡献汇入同一份 `.grad`，不会因为引用了两次就注册两份 hook。

## 6. hook 本体：逐句理解，尤其是 wait

当前代码是：

```python
def _allreduce_gradient(self, parameter: torch.Tensor) -> None:
    if not self._sync_enabled:
        return

    handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
    self.handles.append((parameter, handle))
```

第一句检查开关。如果关闭同步，autograd 已经积累好的梯度留在本地，hook 不做通信。

`all_reduce(..., SUM, async_op=True)` 发起求和，返回一个 `Work` 对象。`async_op=True` 表示可以先拿到任务凭证，不能理解为“返回时 GPU 已经完成求和”。

hook 不等待，也不做除法。它把参数和 handle 存起来，让 backward 继续。`async_op=True` 之所以有用，就是因为发起与等待之间有其他计算可做。

### wait 等的是谁？

这里同时存在两种执行者：

- CPU 执行 Python，给 GPU 提交任务。
- GPU 按 stream（有先后顺序的任务队列）执行任务。

[PyTorch 分布式文档](https://docs.pytorch.org/docs/stable/distributed.html)说明：CPU collective 的 `wait()` 会等待操作完成；通常的 CUDA/NCCL 路径则让当前 CUDA stream 等待通信，CPU 可以继续提交任务。特殊阻塞配置或错误处理路径可能让 host 也等待，不能把“不阻塞 CPU”当作无条件保证。

重点是：**CPU 能继续提交，不等于 GPU 能继续执行。**

如果在 hook 里立即 `wait()`，普通单计算流里后续反向任务会被通信完成依赖限制。旧实现这样写，虽然保证了除法安全，却限制了重叠。

现在的实现把等待推迟到 finish：

```text
hook：C梯度就绪 → 发起C通信 → 保存(parameter, handle) → 返回
backward：继续计算B、A梯度，并在它们就绪时发起通信
finish：等待各任务 → 对对应梯度求平均
optimizer：使用平均梯度更新参数
```

一个 rank 可以先提交某个参数的 all-reduce，但完整结果依赖所有 rank 都为对应 collective 提供梯度；不需要等待所有其他参数的梯度就绪。各 rank 的 collective 必须按兼容的顺序匹配。

**等待和除法要一起放到 finish。** 只移动等待、把除法留在 hook，会修改正在通信的内存，造成数据竞争。发起通信后到收尾前，也不能再次 backward 累加、清空或替换这份 `.grad`。

NCCL 通信通常用独立 stream，finish 中的 `wait()` 建立跨 stream 依赖，让除法晚于求和。Gloo/CPU 下，`wait()` 等待通信完成后，CPU 才做除法。

## 7. 多个 micro-batch：为什么需要 no_sync？

显存放不下大 batch 时，可以分几次 forward/backward，攒好梯度再更新一次。

假设一个 step 分两次，且每次样本数相同：

```text
zero_grad
第1次 backward：累加第1份梯度，不通信
第2次 backward：累加第2份梯度，通信两份梯度的总和
finish
optimizer.step（只更新一次）
```

PyTorch 默认往 `.grad` 累加，所以两次 backward 之间不要 `zero_grad()`。

调用方这样控制：

```python
from contextlib import nullcontext

optimizer.zero_grad(set_to_none=True)
for i in range(N):
    context = model.no_sync() if i < N - 1 else nullcontext()
    with context:
        loss = loss_fn(model(x[i]), y[i]) / N
        loss.backward()

model.finish_gradient_synchronization()
optimizer.step()
```

`loss / N` 和 `grad / world_size` 是两件事：前者平均本地的 N 个等大小 micro-batch，后者平均不同 rank。不能互相替代。

举个具体数值：两次 backward 的未缩放梯度在 rank 0 是 2、4，在 rank 1 是 6、8。

```text
loss / 2 后：rank 0 累积 1 + 2 = 3；rank 1 累积 3 + 4 = 7
最后一次 hook：all-reduce 得到 10，再除以 2，每个 rank 得到 5
这正是四份梯度的平均值：(2 + 4 + 6 + 8) / 4 = 5
```

`no_sync()` 只做一件事：临时把 `_sync_enabled` 设成 False，退出时恢复原值。它不会禁止 backward，也不会清空梯度。`finally` 保证异常退出时仍能恢复；保存原值使嵌套调用也能正确恢复。

前 N−1 次关闭同步，可以避免每个 micro-batch 都通信。对于当前异步实现，也避免上一轮通信尚未收尾时，下一次 backward 往同一份 `.grad` 写入。最后一次发起通信后，必须先 finish，才能开始新的累积周期。

## 8. finish 和 forward：剩下两块其实很简单

`forward` 原样把输入交给 `self.module`，不在这里进行梯度通信。

当前 `finish_gradient_synchronization()`：

```python
for parameter, handle in self.handles:
    handle.wait()
    parameter.grad.div_(self.world_size)
self.handles.clear()
```

在 optimizer 前等待各个任务，对求和后的梯度做原地除法，再清掉凭证列表。`div_` 的下划线表示原地修改。清空的是列表，不是 `.grad`；再次调用 finish 时列表为空，因此不会重复除以进程数。

CUDA 下应理解为：在通常的同一训练计算流上，建立 optimizer 使用梯度前所需的顺序依赖，而不是一定让 Python 等到所有 GPU 工作物理结束。自定义其他 stream 时还要处理相应依赖。

## 9. 为什么 sync 耗时很小，仍不能证明重叠成功？

benchmark 的实际计时顺序是：

```text
计时开始 → forward/backward → synchronize(device)
                            → 开始计 sync → finish → synchronize(device)
                            → 结束计 sync → optimizer.step → synchronize(device)
```

**进入 sync 计时之前，它已经等待设备工作完成。** 对于在 backward hook 中发出的通信，这段等待会记进 `compute_seconds`，而非 `sync_seconds`。

所以即便通信把 backward 卡住，`finish` 本身也可能几乎不耗时。CPU/Gloo 下更直接：等待已经发生在 hook 内，当然不会留很多工作给 finish。

评估性能应看整个 step（此脚本中为 `compute_seconds + sync_seconds`）和 GPU profiler 的计算/通信时间线，不能仅凭收尾计时接近零断言通信被隐藏了。

三个实现可以这样区分：

- `NaiveDDP`：backward 后逐参数同步。调用多，计算与同步分开。
- `FlatDDP`：backward 后把梯度拼起来，一次同步。减少调用次数，但有拼接代价，也没有提前通信。
- 当前 `OverlapDDP`：梯度就绪时逐参数发起异步通信，backward 结束后在 finish 等待并求平均，为计算与通信重叠创造条件。

更完整的 DDP 会把多个参数的梯度放进 bucket，兼顾“早点开始”和“减少小通信次数”。本实现没有 bucket。

## 10. 测试能证明什么？

运行入口（在 `assignment2-systems` 目录中）：

```bash
uv run pytest tests/test_ddp.py
```

当前 `tests/adapters.py` 的 `get_ddp` 指向 `OverlapDDP`，`ddp_on_after_backward` 调用它的 finish。

`tests/test_ddp.py` 用两进程、CPU/Gloo，检查初始化状态对齐，并把分布式训练结果与单进程全 batch 训练比较，连续做 5 次更新；还覆盖普通模型和共享权重模型。

这次数值验证扩展为普通/共享权重模型 × 1/2 个 micro-batch，共 4 个测试，检查 `no_sync()` 不发起通信、finish 清空任务列表、重复 finish 不再次缩放梯度，并与全 batch baseline 比较。2026-10-14 运行 `uv run --offline pytest tests/test_ddp.py`：**4 passed（12.34 秒）**，后端为 CPU/Gloo。

这是数值正确性测试，**不能证明 NCCL/GPU 上真的发生重叠**。本次没有测量 GPU 性能。

## 11. 看累了，只记这条链

```text
初始广播：大家的参数先相同
    ↓
本地 backward：各算各的数据
    ↓
参数 .grad 就绪：触发 post-accumulate hook
    ↓
all-reduce 求和 + 除以进程数：大家拿到同一份平均梯度
    ↓
finish：保证更新前的通信依赖
    ↓
optimizer.step：大家用同一份梯度更新，参数继续一致
```

Overlap 想优化的就是中间那一段：**让已经就绪的梯度通信，与尚未完成的梯度计算同时进行。当前代码用“hook 发起、finish 等待并求平均”的方式，避免在梯度就绪时立即阻挡后续计算。**
