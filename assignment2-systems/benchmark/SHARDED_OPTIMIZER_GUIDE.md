# 优化器状态分片：对照 ShardedOptimizer 理解

实现：[sharded_optimizer.py](../cs336_systems/sharded_optimizer.py)。
测试入口：[adapters.py](../tests/adapters.py) 的 `get_sharded_optimizer`。

## 1. 先分清三个东西

训练里，一个参数不只有权重本身：

- 参数：例如线性层的 `weight`。
- 梯度：`weight.grad`，backward 算出的更新方向。
- 优化器状态：AdamW 为这个参数保存的历史信息，例如一阶矩 `exp_avg`、二阶矩 `exp_avg_sq` 和步数 `step`。

普通 DDP 中，每个 rank 保存完整参数、梯度，也各自维护完整优化器状态。参数和梯度已经相同，AdamW 的历史状态也相同：大家重复做同一份更新。

**这里保留每个 rank 的完整参数与梯度，只分片优化器状态和更新工作。** 这是通常所说的 ZeRO Stage 1 思路，不是参数分片，也不是梯度分片。

假设四个参数分给两个 rank：

```text
参数顺序：      p0    p1    p2    p3
负责的 rank：    0     1     0     1

rank 0：保存所有参数和梯度；仅为 p0、p2 保存优化器状态并更新
rank 1：保存所有参数和梯度；仅为 p1、p3 保存优化器状态并更新
```

每个参数只有一个 owner（负责人）。owner 更新后，把新参数广播给所有人，大家的模型又一致了。

## 2. 和前面 OverlapDDP 的关系

两者同步的对象不同：

```text
backward
    ↓
OverlapDDP：all-reduce 梯度，所有 rank 得到平均梯度
    ↓
ShardedOptimizer：每个 rank 只更新自己负责的参数
    ↓
ShardedOptimizer：广播更新后的参数，所有 rank 得到完整新模型
```

**ShardedOptimizer 自己不平均梯度。** 调用 step 前，调用方必须已经同步梯度。否则 owner 只用自己的本地数据更新，无法得到正确的全局训练结果。

初始化时也要求各 rank 模型参数相同，参数组及参数排列一致。可以由 DDP 的初始广播保证。

标准使用顺序：

```python
from cs336_systems.overlap_ddp import OverlapDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

# 已初始化进程组，且 base_model 已移动到当前 rank 的设备
model = OverlapDDP(base_model)
optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-3)

optimizer.zero_grad()
loss = loss_fn(model(local_x), local_y)
loss.backward()
model.finish_gradient_synchronization()
optimizer.step()
```

梯度累积时仍按 DDP 的规则使用 `no_sync()`，最后一次 backward 后 finish，再调用分片优化器 step。

## 3. 构造函数为什么有两个 Optimizer？

`ShardedOptimizer` 继承 `torch.optim.Optimizer`，内部还包含一个真正做更新的 `_local_optimizer`。

可以把它们理解为：

- 外层：知道所有参数，管理全局参数组、清空所有梯度、同步参数。
- 内层：只收到当前 rank 负责的参数，用 AdamW/SGD 等算法更新这些参数。

必须调用：

```python
super().__init__(params, kwargs)
```

父类建立公开的 `param_groups`、`state` 等 Optimizer 结构，并逐组调用 `self.add_param_group(...)`。由于 Python 的方法分派，这里调用的是我们重写的方法。

所以 `_owners` 和 `_local_optimizer = None` 必须在 `super()` 之前准备好：父类调用 add_param_group 时，需要能访问它们，但此时真正的内层优化器还没构造。

父类初始化结束后：

```python
local_groups = [self._local_group(group) for group in self.param_groups]
self._local_optimizer = optimizer_cls(local_groups, **kwargs)
```

`optimizer_cls` 决定算法，wrapper 不重写 AdamW 公式。传进来的 `lr`、`betas` 等关键字参数原样交给它。

内层初始化后，将它补齐的默认组设置复制给外层，保证公开的参数组也包含完整配置。`self.state` 指向内层状态，使外部能看到当前 rank 实际持有的状态。

## 4. add_param_group：分配所有权，兼容后续加参数

核心规则是：

```python
self._owners[parameter] = len(self._owners) % self.world_size
```

第 0 个唯一参数交给 rank 0，第 1 个交给 rank 1，依此轮流。参数不是按名称匹配，而是根据所有 rank 一致的输入顺序分配。

`_owners` 是“参数对象 → owner 编号”的字典。相同参数对象不会重新分配；共享权重通常在 `model.parameters()` 时已被去重。

每个组交给 `_local_group` 过滤：

```python
"params": [p for p in group["params"] if self._owners[p] == self.rank]
```

它保留该组的学习率等设置，只缩小参数列表。过滤不复制参数数据：内层更新的仍是模型中的同一个 Parameter 对象。

即使某个 rank 在该组没有负责的参数，也保留一个空参数组。这样可以保持全局和本地参数组一一对应；这个 rank 仍需要参与其他 owner 的广播。

训练期间可以添加参数：

```python
optimizer.add_param_group({"params": newly_trainable_parameters, "lr": 1e-4})
```

所有 rank 必须以相同顺序执行这个操作。外层先用父类逻辑验证组，再给新参数分配 owner，最后给已经存在的内层优化器添加对应本地组。原来的 owner 不变。

如果内层拒绝新组，代码会撤销这次外层新增组及所有权，避免留下半完成的分配。

这是一种简单的按参数数量轮流分配策略。参数大小不同时，各 rank 的状态内存不一定均衡；它不会切开单个大参数。

## 5. step：本地更新，然后逐参数广播

step 先把公开参数组中的最新设置传到内层：

```python
for group, local_group in zip(self.param_groups, self._local_optimizer.param_groups):
    for key, value in group.items():
        if key != "params":
            local_group[key] = value
```

为什么每次都复制？scheduler 或用户可能修改外层 `param_groups[0]["lr"]`，内层必须使用新学习率。参数列表不能复制，否则会把全量参数又塞回本地优化器。

接着调用：

```python
result = self._local_optimizer.step(closure=closure, **kwargs)
```

closure 和额外 step 参数交给原优化器处理，并原样返回它的结果。closure 如果重新计算梯度，调用方同样要确保这些梯度已正确同步。

最后：

```python
with torch.no_grad():
    for parameter, owner in self._owners.items():
        dist.broadcast(parameter, src=owner)
```

例如 p1 的 owner 是 rank 1：rank 1 用 p1 的平均梯度和自己保存的 AdamW 状态更新 p1，随后 `broadcast(p1, src=1)`，rank 0 的 p1 被替换为新值。

**这里是广播参数，不是 all-reduce 参数。** 一个 owner 已经算出正确的新参数，不需要再把各 rank 的参数加起来或平均。

所有 rank 都执行完整的广播循环，即使某个 rank 没有负责的参数。`torch.no_grad()` 避免同步写入被 autograd 记录；这里采用同步 collective，保证后续同一训练流可以安全使用更新参数。

## 6. zero_grad 和 checkpoint

外层继承的 `zero_grad()` 遍历完整公开参数组，所以会清空所有梯度。不能只清空本地负责参数的梯度，否则其他参数的旧梯度会留到下一轮。

`state_dict()` 的参数组包含全部参数，但状态只有当前 rank 拥有的部分。`load_state_dict()` 在父类恢复状态后，重新把内层的 `state` 接到外层恢复后的状态，避免内层继续使用旧字典。

这是 **rank 本地 checkpoint**。恢复时要保持相同 world_size、参数顺序及参数组结构，并为每个 rank 恢复对应状态。当前没有跨 rank 合并完整 checkpoint 或改变 world_size 后重新分片的功能。

## 7. 适用范围与边界

wrapper 可以接收不同 Optimizer 类，常见的 AdamW、Adam、SGD 等逐参数维护状态的算法符合这个分片思路。冻结参数或没有梯度的参数是否创建状态、是否更新，由内层优化器决定。

“可以传任意类”不代表所有算法都能保持完整优化器的数学语义。例如依赖全模型联合状态、全局范数或全局线搜索的算法，分开运行本地优化器可能改变结果；LBFGS 也有自己的参数组及线搜索限制。这里没有为这些算法实现额外的跨 rank 协调。

该实现也没有通信打包或异步参数广播。每个参数一次 broadcast，主要目标是清楚地实现作业要求的状态分片。

## 8. 测试验证什么？

运行（在 assignment2-systems 目录）：

```bash
uv run pytest tests/test_sharded_optimizer.py
```

原有两个测试用两进程 CPU/Gloo，分别检查普通模型与共享权重模型。各 rank 使用相同输入和初始权重，连续训练 10 步，比较分片 AdamW 与普通 AdamW 的最终参数；因此测试本身不需要额外 DDP 梯度同步。

新增参数组测试覆盖：某个 rank 初始没有本地参数、动态添加组、不同组学习率、修改公开学习率、closure 返回值、跨 rank 参数一致、本地只保存负责参数的状态，以及本地 checkpoint 恢复后继续训练。

2026-10-14 连续运行 5 轮，每轮 **3 passed**，共 15 次测试全部通过（每轮约 3.1–3.4 秒）。CPU 正确性测试不代表 GPU 上的实际内存节省或通信性能已经测量。

## 9. 最后记住这一句

**梯度先同步；参数由 owner 更新；状态只在 owner 保存；更新后的参数再广播给所有人。**

分片省的是重复的优化器历史状态，并让各 rank 分担更新工作；它没有让任何 rank 丢掉完整模型。
