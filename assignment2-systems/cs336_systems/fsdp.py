"""按层分片 Linear/Embedding；计算时拼齐，优化器只更新 FP32 分片。

阅读顺序：FSDP 初始化 → _LayerForward.__call__ → _LayerCall.forward/backward
→ finish_gradient_synchronization。
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from cs336_basics.model import Embedding, Linear


@dataclass
class _Shard:
    """还原说明：原形状、真实元素数、每卡分片长度。"""

    original_shape: torch.Size
    original_numel: int
    shard_size: int


@dataclass
class _GradientTask:
    """一次梯度通信：给谁写梯度、结果在哪、通信凭证是什么。"""

    parameter: nn.Parameter
    result: torch.Tensor
    work: dist.Work | None
    send_buffer: torch.Tensor | None
    is_sharded: bool

    def wait(self):
        if self.work is not None:
            self.work.wait()
            self.work = None
        # 完整发送缓冲不再保留，只留下接收结果。
        self.send_buffer = None


class _LayerForward:
    """代替某层的 forward；这是普通可调用对象，不是新子模块。"""

    def __init__(self, fsdp, layer, parameter_names):
        self.fsdp = fsdp
        self.layer = layer
        self.parameter_names = parameter_names
        self.original_forward = layer.forward

        # 四种支持的层都是单输入，但原函数的输入名称不同。
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            self.input_name = "input"
        elif isinstance(layer, Linear):
            self.input_name = "x"
        else:
            self.input_name = "token_ids"

    def __call__(self, *inputs, **kwargs):
        # layer(x) 与 layer(input=x) 都取到同一个输入 tensor。
        if inputs:
            if len(inputs) != 1 or kwargs:
                raise TypeError("Linear/Embedding expects exactly one input tensor.")
            x = inputs[0]
        else:
            if len(kwargs) != 1 or self.input_name not in kwargs:
                raise TypeError(f"Expected one input tensor named {self.input_name!r}.")
            x = kwargs[self.input_name]

        local_parameters = []
        for name in self.parameter_names:
            parameter = self.layer._parameters[name]
            local_parameters.append(parameter)

        return _LayerCall.apply(self, x, *local_parameters)

    def run_original_forward(self, full_weights, x):
        """借完整权重给原层，执行完立刻把分片装回去。"""
        saved_parameters = {}
        for name in self.parameter_names:
            saved_parameters[name] = self.layer._parameters[name]

        try:
            for name, full_weight in zip(self.parameter_names, full_weights):
                self.layer._parameters[name] = full_weight

            is_linear = isinstance(self.layer, (nn.Linear, Linear))
            if is_linear and self.fsdp.compute_dtype is not None:
                x = x.to(self.fsdp.compute_dtype)

            return self.original_forward(x)
        finally:
            for name, parameter in saved_parameters.items():
                self.layer._parameters[name] = parameter


class _LayerCall(torch.autograd.Function):
    """前向不保存完整权重；反向重新收集并重算当前这一层。"""

    @staticmethod
    def forward(ctx, layer_forward, x, *local_parameters):
        ctx.layer_forward = layer_forward
        ctx.local_parameters = local_parameters
        ctx.save_for_backward(x)

        full_weights = layer_forward.fsdp._gather_weights(local_parameters)
        output = layer_forward.run_original_forward(full_weights, x)
        # ctx 只存输入和本地 Parameter，不存 full_weights。
        return output

    @staticmethod
    def backward(ctx, grad_output):
        layer_forward = ctx.layer_forward
        fsdp = layer_forward.fsdp
        x = ctx.saved_tensors[0]
        full_weights = fsdp._gather_weights(ctx.local_parameters)

        # apply 的第 0 项是 layer_forward，第 1 项是 x。
        needs_input_gradient = ctx.needs_input_grad[1]

        with torch.enable_grad():
            local_x = x.detach()
            local_x.requires_grad_(needs_input_gradient)

            differentiable_weights = []
            for full_weight, parameter in zip(full_weights, ctx.local_parameters):
                weight = full_weight.detach()
                weight.requires_grad_(parameter.requires_grad)
                differentiable_weights.append(weight)

            # 只重算当前层，不重算整个模型或 loss。
            output = layer_forward.run_original_forward(differentiable_weights, local_x)

            gradient_targets = []
            if needs_input_gradient:
                gradient_targets.append(local_x)
            for weight in differentiable_weights:
                if weight.requires_grad:
                    gradient_targets.append(weight)

            gradients = torch.autograd.grad(
                outputs=output,
                inputs=gradient_targets,
                grad_outputs=grad_output,
                allow_unused=True,
            )

        gradient_index = 0
        grad_x = None
        if needs_input_gradient:
            grad_x = gradients[gradient_index]
            gradient_index += 1

        for parameter in ctx.local_parameters:
            if not parameter.requires_grad:
                continue
            full_gradient = gradients[gradient_index]
            gradient_index += 1
            if full_gradient is not None:
                fsdp._start_sharded_gradient_sync(parameter, full_gradient)

        # 和 apply(layer_forward, x, *local_parameters) 逐项对应。
        # 分片梯度还在通信，等 finish 后再写入 parameter.grad。
        input_gradients = [None, grad_x]
        for parameter in ctx.local_parameters:
            input_gradients.append(None)
        return tuple(input_gradients)


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize the distributed process group before creating FSDP.")
        allowed_dtypes = (None, torch.float32, torch.float16, torch.bfloat16)
        if compute_dtype not in allowed_dtypes:
            raise ValueError("compute_dtype must be a floating-point training dtype.")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._shards: dict[nn.Parameter, _Shard] = {}
        self._gradient_tasks: list[_GradientTask] = []

        # 第一步：让所有 rank 从相同的 FP32 参数开始。
        self._broadcast_initial_model()

        # 第二步：找到目标层，切分参数，再替换 forward 入口。
        supported_layers = (nn.Linear, nn.Embedding, Linear, Embedding)
        for layer in module.modules():
            if not isinstance(layer, supported_layers):
                continue
            if isinstance(layer, nn.Embedding):
                if layer.max_norm is not None or layer.sparse:
                    raise ValueError("FSDP does not support max_norm or sparse Embedding.")

            parameter_names = []
            for name, parameter in layer.named_parameters(recurse=False):
                parameter_names.append(name)
                self._shard_parameter(parameter)

            # layer(x) 会调用该对象的 __call__，不需要 MethodType/inspect。
            layer.forward = _LayerForward(self, layer, parameter_names)

        # 第三步：RMSNorm 等未切分参数仍需同步完整梯度。
        for parameter in module.parameters():
            if parameter in self._shards:
                continue
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self._start_replicated_gradient_sync)

    @torch.no_grad()
    def _broadcast_initial_model(self):
        for parameter in self.module.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()
            dist.broadcast(parameter, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer, src=0)

    @torch.no_grad()
    def _shard_parameter(self, parameter):
        # 共享权重是同一个 Parameter，只切一次。
        if parameter in self._shards:
            return

        original_shape = parameter.shape
        original_numel = parameter.numel()
        shard_size = (original_numel + self.world_size - 1) // self.world_size
        padded_size = shard_size * self.world_size

        padded = torch.zeros(padded_size, dtype=torch.float32, device=parameter.device)
        original_values = parameter.detach().reshape(-1)
        padded[:original_numel].copy_(original_values)

        start = self.rank * shard_size
        end = start + shard_size
        # clone 避免切片继续引用完整 padded 存储。
        parameter.data = padded[start:end].clone()
        self._shards[parameter] = _Shard(original_shape, original_numel, shard_size)

    def _wait_for_gradient_communications(self):
        # 撤掉上一层完整梯度的发送缓冲，避免逐层堆积。
        # 结果分片还在 task.result，finish 再求平均、写入 .grad。
        for task in self._gradient_tasks:
            task.wait()

    def _gather_weights(self, local_parameters):
        self._wait_for_gradient_communications()
        full_weights = []
        communication_dtype = self.compute_dtype or torch.float32

        for parameter in local_parameters:
            shard_info = self._shards[parameter]
            local_values = parameter.detach().to(communication_dtype).contiguous()

            received_shards = []
            for rank in range(self.world_size):
                received_shards.append(torch.empty_like(local_values))

            work = dist.all_gather(received_shards, local_values, async_op=True)
            # 当前层马上要用完整权重，必须在计算前等待。
            work.wait()

            padded_values = torch.cat(received_shards)
            original_values = padded_values[:shard_info.original_numel]
            full_weight = original_values.view(shard_info.original_shape)
            full_weights.append(full_weight)

        return full_weights

    def _start_sharded_gradient_sync(self, parameter, full_gradient):
        shard_info = self._shards[parameter]
        padded_size = shard_info.shard_size * self.world_size
        send_buffer = torch.zeros(padded_size, dtype=torch.float32, device=parameter.device)
        send_buffer[:shard_info.original_numel].copy_(full_gradient.detach().reshape(-1))
        local_gradient = torch.empty_like(parameter)

        work = dist.reduce_scatter_tensor(
            output=local_gradient,
            input=send_buffer,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        task = _GradientTask(
            parameter=parameter,
            result=local_gradient,
            work=work,
            send_buffer=send_buffer,
            is_sharded=True,
        )
        self._gradient_tasks.append(task)

    def _start_replicated_gradient_sync(self, parameter):
        full_gradient = parameter.grad.detach().clone()
        work = dist.all_reduce(full_gradient, op=dist.ReduceOp.SUM, async_op=True)
        task = _GradientTask(
            parameter=parameter,
            result=full_gradient,
            work=work,
            send_buffer=full_gradient,
            is_sharded=False,
        )
        self._gradient_tasks.append(task)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        for task in self._gradient_tasks:
            task.wait()
            average_gradient = task.result
            average_gradient.div_(self.world_size)

            parameter = task.parameter
            if task.is_sharded and parameter.grad is not None:
                # 共享参数多次使用时，累加各次的梯度贡献。
                parameter.grad.add_(average_gradient)
            else:
                parameter.grad = average_gradient

        self._gradient_tasks.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """检查/导出用：收集完整 FP32 参数，不永久改变训练分片。"""
        self.finish_gradient_synchronization()
        full_parameters = {}
        already_gathered = {}

        for name, parameter in self.module.named_parameters(remove_duplicate=False):
            if parameter in already_gathered:
                full_parameters[name] = already_gathered[parameter]
                continue

            if parameter in self._shards:
                shard_info = self._shards[parameter]
                received_shards = []
                for rank in range(self.world_size):
                    received_shards.append(torch.empty_like(parameter))
                dist.all_gather(received_shards, parameter.detach())

                padded_values = torch.cat(received_shards)
                original_values = padded_values[:shard_info.original_numel]
                full_weight = original_values.view(shard_info.original_shape)
            else:
                full_weight = parameter.detach().clone()

            already_gathered[parameter] = full_weight
            full_parameters[name] = full_weight

        return full_parameters
