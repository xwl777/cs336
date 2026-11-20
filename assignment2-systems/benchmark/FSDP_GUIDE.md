# FSDP：拿一个只有 4 个数的模型，把整个过程走一遍

对照代码：[fsdp.py](../cs336_systems/fsdp.py)。

这篇先不要求你记住 hook、ctx、autograd。我们只回答三个问题：**每张卡平时存什么？计算时缺了东西怎么办？算完怎么把结果分回去？** 等这个过程明白了，再去看代码为什么要这么写。

后半部分按重构后的函数逐句展开。第一次读先看前半部分的小例子，遇到 Python 细节再往后查，不用一次读完。

## 1. 先拿一个真正能算出来的小层

假设整个模型暂时只有一个 Linear 层，没有 bias。它把两个输入数变成两个输出数，权重是：

```text
W = [[1, 2],
     [3, 4]]
```

输入 `x = [1, 2]` 时，输出怎么算？

```text
第一个输出：1×1 + 2×2 = 5
第二个输出：1×3 + 2×4 = 11
所以 y = [5, 11]
```

这里只要理解：**计算这两个输出，需要 W 中的四个数。** 少一半，就没法直接照这个普通 Linear 公式算出全部输出。

别把“一个参数”和“一个数”混起来。`layer.weight` 是一个 Parameter 对象，里面装着整个 `[2, 2]` 的 W。这次 FSDP 会把这个对象的数据拆开，让各 rank 只留下其中几个数。

我们用两张卡，rank 0 就是第 0 个进程，rank 1 就是第 1 个进程。通常一个进程负责一张卡。

## 2. 原来是两个人各存一整本，现在每人只存半本

普通 DDP 的初始化结果：

```text
rank 0 保存：[[1, 2], [3, 4]]
rank 1 保存：[[1, 2], [3, 4]]
```

两张卡各有完整 W。之前的 ShardedOptimizer 也仍这样保存 W，它只是把 AdamW 的历史状态分开了。

这次 FSDP 初始化后变成：

```text
先把 W 拉成一条：[1, 2, 3, 4]

rank 0 保存：[1, 2]
rank 1 保存：[3, 4]
```

这就是 shard：一段数据，中文叫分片。

**它和前面的 ShardedOptimizer 分配方法不同。** 前面是一个完整 tensor 交给一个 owner；这里是把同一个 tensor 内部的元素切开，让所有 rank 各持一段。

在这个例子里正好一张卡存一行。实际代码按扁平元素切，分片边界不一定刚好落在矩阵行末。

这一步能省常驻权重内存，但立刻带来问题：rank 0 只有 `[1, 2]`，它怎么计算第二个输出？

答案是：**需要计算这一层时，临时把另一半拿回来。**

## 3. forward：临时借另一半，算完就还

执行 Linear 之前，两张卡做 all-gather。

`gather` 是收集，`all` 是每个人都收到：把各自那一份收集到所有 rank。

```text
收集前：
rank 0：[1, 2]
rank 1：[3, 4]

收集后，各自临时得到：
rank 0：[1, 2, 3, 4]
rank 1：[1, 2, 3, 4]
```

然后恢复矩阵形状：`[[1, 2], [3, 4]]`。

两张卡读不同的数据，但都用同一个完整 W：

```text
rank 0 的输入：[1, 2] → 输出 [5, 11]
rank 1 的输入：[3, 4] → 输出 [11, 25]
```

**这里仍是数据并行。** 不是 rank 0 算第一行输出、rank 1 算第二行输出；两边都算完整的层，只是处理不同数据。

这一层 forward 结束后，临时完整 W 不再保存，平时留下的又是：

```text
rank 0 的 master：[1, 2]
rank 1 的 master：[3, 4]
```

`master` 就是 optimizer 真正更新的那份权重。本实现让它始终保持 FP32。

### 临时拿回完整 W，那不还是不省内存吗？

如果整个模型只有这一层，执行它时确实仍要装下完整 W，还会有通信临时缓冲，峰值未必好看。

但真实模型有很多层。假设有 A、B、C 三层：

```text
算 A：只临时拼完整 A；B、C 还是分片
算 B：A 的完整副本已撤掉；只临时拼完整 B
算 C：B 的完整副本已撤掉；只临时拼完整 C
```

省的是：**不用让所有层的完整权重一直同时住在每张卡上。** 它不是让正在计算的这一层也永远只用半个矩阵。

本实现按层实际被调用时才收集，没有预取未来层。这比题目的提前收集上限更保守，但通信隐藏效果也有限。

## 4. backward：为什么又要把刚扔掉的 W 拿回来？

Linear 的 backward 有两件工作，别混成一个：

1. 算 W 的梯度：告诉 optimizer，权重该往哪边改。
2. 算 x 的梯度：告诉前面那一层，它的输出该往哪边改。

为了看懂，先设这个小例子的 loss 为 `y 的两个数相加`。因此：

```text
loss = y0 + y1
loss 对 y 的梯度：[1, 1]
```

`grad_output` 就是这份“loss 对这一层输出的梯度”。真实模型中它从后面的层传回来，不一定是 `[1, 1]`。

拿 rank 0 的 `x=[1, 2]` 来看：

```text
y0 = 1×x0 + 2×x1
y1 = 3×x0 + 4×x1
```

要算输入梯度，需要用到 W：

```text
loss 对 x0 的梯度：1 + 3 = 4
loss 对 x1 的梯度：2 + 4 = 6
所以 grad_x = [4, 6]
```

这就是为什么 backward 通常还需要完整 W。前向时为了省内存扔掉了，现在重新 all-gather。

W 的梯度则来自这一层的输入与传回来的输出梯度：

```text
rank 0 输入 [1, 2]，本地 W 梯度：[[1, 2], [1, 2]]
rank 1 输入 [3, 4]，本地 W 梯度：[[3, 4], [3, 4]]
```

两边 W 相同，输入不同，所以本地 W 梯度不同。这完全正常，还没有做跨 rank 平均。

这个“求输出和”的 loss 只是方便手算，不是在说真实训练一定用这种 loss。假设每卡一个样本，后面取两张卡的平均，等价于这两个样本的平均 loss 梯度。

## 5. reduce-scatter：先把梯度加起来，再每人拿自己那一段

先把上面的梯度拉平：

```text
rank 0：[1, 2, 1, 2]
rank 1：[3, 4, 3, 4]
```

逐位置求和，得到：

```text
[4, 6, 4, 6]
```

但各 rank 只更新自己保存的权重片段，所以不需要大家拿到完整梯度：

```text
rank 0 收到前半：[4, 6]
rank 1 收到后半：[4, 6]
```

这就是 reduce-scatter：`reduce` 求和，`scatter` 分发不同部分。

当前代码用的是 SUM，所以 finish 还要除以 2：

```text
rank 0 的平均梯度：[2, 3]，对应权重 [1, 2]
rank 1 的平均梯度：[2, 3]，对应权重 [3, 4]
```

**权重片段和梯度片段必须对应同一组元素。** rank 0 不能拿着后半的梯度更新前半权重。

还要分清：每张卡的 backward 仍会计算完整的本地 W 梯度，只是通信完成后只长期保留归约后的本地分片。并不是“不负责的那几个数就不用求导了”。

## 6. optimizer：每张卡只改自己保存的数

为了能手算，先用学习率 0.1 的 SGD：

```text
rank 0：
[1, 2] - 0.1 × [2, 3] = [0.8, 1.7]

rank 1：
[3, 4] - 0.1 × [2, 3] = [2.8, 3.7]
```

现在两张卡各自保存的是新权重的一部分。

这里 **不需要立刻再广播整个 W**。下一次 forward 用到这个 Linear 时，all-gather 就会得到：

```text
W_new = [[0.8, 1.7],
         [2.8, 3.7]]
```

如果用 AdamW，更新公式不同，但分工一样：rank 0 为 `[1, 2]` 那两个元素维护 m、v，rank 1 为 `[3, 4]` 那两个元素维护 m、v。普通 AdamW 只看到本地分片，因此状态自然也是分片的，不需要它知道整个模型 W 长什么样。

**为什么先包 FSDP，再创建 optimizer？** 因为 optimizer 应该从一开始就看见“更新对象是本地 FP32 分片”。别在完整权重上建立好状态后，再把权重偷偷切开。

到这里，完整链路已经走完：

```text
各存半份 W
  → forward 前拼 W
  → 各算各的数据，算完撤掉完整 W
  → backward 前再次拼 W
  → 算完整本地梯度，归约并切回本地梯度
  → finish 等通信并求平均
  → optimizer 只更新本地 W
```

下面才开始解释代码里的那些“绕”的地方。

## 7. 新代码先看四个对象，各管一件事

`FullyShardedDataParallel` 是总负责人：切参数、发起通信、收尾写梯度。`_LayerForward` 代表一个被包装的层，记住原来的 forward 和这一层有哪些参数。`_LayerCall` 是 forward/backward 这一对计算规则。`_GradientTask` 代表一笔还没完成的梯度通信。

不要一下子把它们都看完。按下面顺序读，能跟着一次 `model(x)` 走到底。

## 8. 初始化：先找层，再切参数，最后改入口

构造函数的核心现在直接写成：

```python
self._broadcast_initial_model()

for layer in module.modules():
    if not isinstance(layer, supported_layers):
        continue

    parameter_names = []
    for name, parameter in layer.named_parameters(recurse=False):
        parameter_names.append(name)
        self._shard_parameter(parameter)

    layer.forward = _LayerForward(self, layer, parameter_names)
```

逐句读：

1. 所有人先用 rank 0 的初始模型，避免权重起点不同。
2. `module.modules()` 遍历模型本身和内部所有子模块。
3. 只挑 Linear、Embedding 处理，其他模块跳过。
4. `named_parameters(recurse=False)` 只取当前层自己的 weight、bias，不再进入子层。
5. 记录这些名字，并逐个切分参数。
6. 把这一层的 forward 换成 `_LayerForward` 对象。

`self` 是 FSDP 总负责人；`layer` 是正在处理的具体 Linear/Embedding；`parameter_names` 通常是 `["weight"]` 或 `["weight", "bias"]`。

原生 `nn.Linear/nn.Embedding` 和作业 1 的 `Linear/Embedding` 都在支持名单。原生 Embedding 如果启用 sparse 或 max_norm，会明确报错，因为这需要额外实现。

## 9. _shard_parameter：这次把切片写直白了

```python
original_shape = parameter.shape
original_numel = parameter.numel()
shard_size = (original_numel + self.world_size - 1) // self.world_size
padded_size = shard_size * self.world_size

padded = torch.zeros(padded_size, dtype=torch.float32, device=parameter.device)
original_values = parameter.detach().reshape(-1)
padded[:original_numel].copy_(original_values)

start = self.rank * shard_size
end = start + shard_size
parameter.data = padded[start:end].clone()
```

带入四数例子：

```text
original_shape = [2, 2]
original_numel = 4
shard_size = 2
padded_size = 4

先创建 padded：[0, 0, 0, 0]
填入原数据：  [1, 2, 3, 4]

rank 0：start=0，end=2 → [1, 2]
rank 1：start=2，end=4 → [3, 4]
```

`detach()` 表示复制、分片这件事不需要求导，`reshape(-1)` 表示展平。现在不再用 `detach().float().reshape(-1).new_zeros(...)` 连着绕一圈创建零数组，而是直接 `torch.zeros(...)`。

如果 5 个数给 2 张卡，每卡长度是 3，总长度 6，最后补一个 0。拼回时只保留真实的前 5 个数。

`.clone()` 的作用是分配独立的小存储。如果只留 `padded[start:end]` 的 view，它仍可能引用整个大 padded，虽然看起来只有半份，原来的完整存储却不能释放。

随后 `_Shard(original_shape, original_numel, shard_size)` 记住还原说明。共享权重若是同一个 Parameter，`_shards` 查到已有记录就跳过，不会把已经切好的分片再切一次。

## 10. 终于没有 MethodType：为什么一个对象能当 forward？

现在只需要：

```python
layer.forward = _LayerForward(self, layer, parameter_names)
```

先理解一个与 PyTorch 无关的小例子：

```python
class Double:
    def __call__(self, x):
        return x * 2

f = Double()
y = f(3)  # 实际运行 f.__call__(3)，得到 6
```

对象有 `__call__`，就能像函数一样调用。

所以把 `_LayerForward` 对象赋给 `layer.forward` 后，`layer(x)` 正常进入 PyTorch 的层调用逻辑，再调用这个对象的 `__call__`。对象把需要的信息保存成普通属性，不需要多层闭包，也不需要自动绑定 `this`。

看看它构造时保存了什么：

```python
self.fsdp = fsdp                         # 找谁做通信
self.layer = layer                       # 当前哪一层
self.parameter_names = parameter_names   # weight、bias 等名字
self.original_forward = layer.forward   # 保存旧函数，此时还没替换
```

这里的 `self` 是 `_LayerForward` 对象自身。`self.fsdp` 才是整个 FSDP，`self.layer` 才是原 Linear。名字写出来后，不用再区分闭包里的 owner/this。

### __call__ 先把输入拿出来

`layer(x)` 的 x 在 `inputs[0]`；`layer(input=x)` 的 x 在 `kwargs["input"]`。

四种支持的层都只有一个输入，但名字不同：原生层叫 input，作业 Linear 叫 x，作业 Embedding 叫 token_ids。在构造时用普通 if/elif 记住名称，调用时按这个名字取，不再解析函数签名。传多个输入或错误名字会报 TypeError。

然后取当前层的本地参数：

```python
local_parameters = []
for name in self.parameter_names:
    parameter = self.layer._parameters[name]
    local_parameters.append(parameter)
```

如果名字是 `["weight", "bias"]`，结果就是 `[本地 weight 分片, 本地 bias 分片]`。

最后交给自定义节点：

```python
return _LayerCall.apply(self, x, *local_parameters)
```

`*local_parameters` 表示展开列表。若有两个参数，就相当于传入 `self, x, weight, bias`。这里传 self，是让计算节点知道回头找哪个层、哪个 FSDP。

## 11. _LayerCall.forward：拼齐，借给原层，只记住输入

```python
ctx.layer_forward = layer_forward
ctx.local_parameters = local_parameters
ctx.save_for_backward(x)

full_weights = layer_forward.fsdp._gather_weights(local_parameters)
output = layer_forward.run_original_forward(full_weights, x)
return output
```

`ctx` 是这一次调用的记事本。前向写进去，反向再拿出来；每次层调用都有自己的记事本。

保存的是入口对象、本地 Parameter 和输入 x。**不保存完整 full_weights。** `_gather_weights` 负责借齐权重，`run_original_forward` 负责用完整权重运行原层。

为什么用自定义 Function？因为仅把 `layer.weight` 换回分片，不保证完整 W 真被释放：普通 autograd 可能还保存着它，准备 backward 使用。自定义 forward 不建立内部普通逐运算图，所以这里可以明确不让前向图长期留住 full_weights。

输入 x 仍保存，因此本实现没有把激活内存也完全消除。

## 12. _gather_weights：一行只讲一件事

```python
local_values = parameter.detach().to(communication_dtype).contiguous()

received_shards = []
for rank in range(self.world_size):
    received_shards.append(torch.empty_like(local_values))

work = dist.all_gather(received_shards, local_values, async_op=True)
work.wait()

padded_values = torch.cat(received_shards)
original_values = padded_values[:shard_info.original_numel]
full_weight = original_values.view(shard_info.original_shape)
```

rank 0 的 local_values 是 `[1, 2]`。received_shards 先准备两个空接收位置；all-gather 后是 `[[1, 2], [3, 4]]`。cat 连成 `[1, 2, 3, 4]`，取真实元素、恢复原矩阵形状。

如果指定 FP16/BF16，local_values 在通信前就转换精度，而 master Parameter 不变。这里 cat 还会短暂创建完整拼接副本，因此仍有当前层级别的通信临时内存。

为什么这里立即 wait？**当前层没有完整 W 就算不了。** 这与 OverlapDDP 中“后续层不依赖当前参数梯度的同步结果”不同，按需收集方案不能把这次等待直接推迟到整个 forward 之后。

函数开始还会先等待之前的梯度任务，释放上一层完整梯度的发送缓冲，避免所有层的发送数据逐个堆到 backward 末尾。

## 13. run_original_forward：先换完整权重，算完必须换回来

```python
saved_parameters = {}
for name in self.parameter_names:
    saved_parameters[name] = self.layer._parameters[name]

try:
    for name, full_weight in zip(self.parameter_names, full_weights):
        self.layer._parameters[name] = full_weight
    return self.original_forward(x)
finally:
    for name, parameter in saved_parameters.items():
        self.layer._parameters[name] = parameter
```

按时间顺序读：记住原分片 → 临时让原层读完整权重 → 执行原 forward → 无论成功还是报错，都把原分片装回来。

optimizer 引用的 master Parameter 身份不变，只是执行原层的那一瞬间让层入口读临时完整 tensor。源码还在 Linear 计算前把浮点输入转换成指定计算精度；Embedding 的 token ID 是整数，不能做这个转换。

## 14. backward：拿回输入，重建当前层的小图

开头把名字写全：

```python
layer_forward = ctx.layer_forward
fsdp = layer_forward.fsdp
x = ctx.saved_tensors[0]
full_weights = fsdp._gather_weights(ctx.local_parameters)
```

就是取记事本、找 FSDP、取原输入、再次拼权重。

接着：

```python
needs_input_gradient = ctx.needs_input_grad[1]
```

apply 的参数从 0 编号：第 0 个是 layer_forward，第 1 个是 x。这句问 PyTorch：前面的图需不需要 x 的梯度？如果是整数 token ID，就不需要；Embedding 的权重仍然需要求导。

`with torch.enable_grad()` 里面创建与旧图脱开的 local_x 和完整权重叶子 tensor，然后重算当前层。detach 是为了让这段局部求导不沿旧图跑回整个模型。

求导目标明确逐个添加：

```python
gradient_targets = []
if needs_input_gradient:
    gradient_targets.append(local_x)
for weight in differentiable_weights:
    if weight.requires_grad:
        gradient_targets.append(weight)
```

若有 x、weight、bias 三项，需要的结果顺序就是 `[grad_x, grad_weight, grad_bias]`。如果 x 不求导，则只列权重；冻结参数也不列进去。

然后：

```python
gradients = torch.autograd.grad(
    outputs=output,
    inputs=gradient_targets,
    grad_outputs=grad_output,
    allow_unused=True,
)
```

意思是：后面已经传回 grad_output，请求这一层输入和权重的梯度。**不是重算整个模型，不是重新求 loss。** 只是用重新建的当前层小图求导。

后面用 gradient_index 从 0 开始顺序取结果。若需要 grad_x，先取一个，索引加 1；再逐个取可训练权重的完整梯度，送入 `_start_sharded_gradient_sync`。

最后构造返回值：

```python
input_gradients = [None, grad_x]
for parameter in ctx.local_parameters:
    input_gradients.append(None)
return tuple(input_gradients)
```

与 apply 的输入逐项对应：layer_forward 不是 tensor，返回 None；x 的梯度还给上一层；权重分片的梯度尚在通信，不通过默认返回路径累加，留到 finish 写 `.grad`。

这是新增一次当前层 forward 换内存，不代表所有 FSDP 都必须重算。这里没有实现高阶梯度或带随机/状态副作用的目标层重算。

## 15. _GradientTask：不再猜五元组每项是什么

以前 `_pending` 里每项有 5 个位置，不看定义很难知道谁是谁。现在写成：

```python
task = _GradientTask(
    parameter=parameter,
    result=local_gradient,
    work=work,
    send_buffer=send_buffer,
    is_sharded=True,
)
self._gradient_tasks.append(task)
```

分别表示：最后给哪个 Parameter 写梯度，收到的梯度放在哪里，通信凭证是什么，正在发送的数据在哪里，以及它是不是分片参数。

`_start_sharded_gradient_sync` 把完整本地梯度补零到可均分尺寸，再启动 reduce-scatter。发送及归约使用 FP32，结果形状与 FP32 master 分片一致。

`_start_replicated_gradient_sync` 用于 RMSNorm 等未分片参数：克隆完整梯度，再发起 all-reduce。

两种任务都提供 `task.wait()`：如果 work 尚未等待，就 wait 并清掉凭证；之后撤掉发送缓冲引用。结果仍保留在 task.result。这样调用者不再手动替换整个五元组。

## 16. finish：把任务结果写进 optimizer 要读的格子

```python
for task in self._gradient_tasks:
    task.wait()
    average_gradient = task.result
    average_gradient.div_(self.world_size)

    parameter = task.parameter
    if task.is_sharded and parameter.grad is not None:
        parameter.grad.add_(average_gradient)
    else:
        parameter.grad = average_gradient

self._gradient_tasks.clear()
```

按顺序：等通信 → SUM 除以卡数 → 找参数 → 写入 `.grad`。如果同一共享权重被多次使用，各次分片梯度需要加起来。

未分片参数的 hook 读取的是累计 `.grad`，因此它使用赋值，而不是再把累计值重复加一遍。

清空任务后，再次 finish 不会再次除法。Gloo 的 wait 等 CPU 通信完成；NCCL 通常是在当前 GPU 流建立依赖，让之后的求平均和更新晚于通信。

## 17. gather_full_params：只为检查，不是训练时常驻

它逐个收集完整 FP32 参数，去掉 padding、恢复形状，并用原模型名字返回字典。共享 Parameter 只收集一次，但可以对应多个名字。

这不改变模型中平时的分片，也不会让 optimizer 变成更新完整权重。但返回字典本身占完整模型存储，因此只在检查/导出时使用；它不是包含全部 buffer 的完整 checkpoint 加载方案。

## 18. 计算精度与使用边界

平时 master 分片、finish 写入的梯度和对应 AdamW 状态保持 FP32；指定 compute_dtype 时，权重通信与临时计算权重使用 FP16/BF16，而梯度归约仍是 FP32。没有梯度缩放器，不承诺任意 FP16 训练都稳定。

所有 rank 需要一致的模型与层执行路径，按相同顺序参与 collective。先包装，再创建 optimizer；backward 后必须 finish，再 step。不要叠加之前的 ShardedOptimizer 或 DDP wrapper。

当前层实际调用时才收集，不提前预取未来层；未分片的小参数用完整 all-reduce。没有通信打包、no_sync、并发调用、高阶求导、稀疏 Embedding 或 max_norm Embedding 支持。

任意嵌套 nn.Module 可以承载目标层，但目标层本身按原生/作业的标准单输入接口调用。自定义子类若改了参数名或加入额外输入，需要扩展 `_LayerForward` 的输入处理。直接拿已经分片的 weight 在外部做完整矩阵运算也需要另行处理。

## 19. 验证

```bash
# 在 assignment2-systems 目录
uv run pytest tests/test_fsdp.py
```

adapter 已接入 get_fsdp、fsdp_on_after_backward、fsdp_gather_full_params。覆盖 FP32/FP16 训练等价性、梯度同步、原生层、共享权重、冻结 bias、padding、关键字输入、重复 finish，以及作业 1 AdamW。

2026-10-15 可读性重构后连续运行五轮，每轮 5 passed，共 25 次测试通过；各轮耗时为 12.42、6.13、7.89、8.60、12.59 秒。测试使用 CPU/Gloo；当前机器没有 CUDA GPU，没有 GPU/NCCL 性能、显存峰值或 BF16 实测。
