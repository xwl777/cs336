import torch
import argparse
import os
import json
import yaml
import typing
import time
import numpy as np
from collections.abc import Callable
import random
import swanlab
import math

from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.data_loader import *
from cs336_basics.learning_rate import *
from cs336_basics.gradient_clip import *
from cs336_basics.loss import *
from cs336_basics.checkpoint import * 

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

def load_data(dataset:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    return np.memmap(dataset, dtype = np.uint16, mode="r")

def load_config(config_path:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    ext = os.path.splitext(config_path)[1].lower()
    if ext == ".json":
        with open(config_path) as f:
            return Config(json.load(f))
    elif ext in [".yaml", ".yml"]:
        with open(config_path) as f:
            return Config(yaml.safe_load(f))
    else:
        raise ValueError(f"config file type{ext} is not supported")
    
class Logger():
    def __init__(self, log_file:str | os.PathLike | typing.BinaryIO | typing.IO[bytes] | None = None, swanlab_run:swanlab.Run | None = None, resume:bool = False):
        self.log_file = log_file
        self.swanlab_run = swanlab_run
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok = True)
            if not resume:
                with open(log_file, "w"):
                    pass

    def format_message(self, metrics:dict):
        message = list()
        for k, v in metrics.items():
            message.append(f"{k}:{v}")

        return " | ".join(message)

    def log_info(self, message:str | dict, to_console = True):
        if isinstance(message, dict):
            message = self.format_message(message)

        if to_console:
            print(message)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_swanlab(self, metrics:dict):
        if self.swanlab_run:
            self.swanlab_run.log(metrics)


    
def setup_seed(seed:int, using_cuda:bool=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if using_cuda:
        torch.cuda.manual_seed_all(seed)

def setup_rng_state(rng_state:dict, using_cuda:bool=True):
    torch.set_rng_state(rng_state["torch"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])
    if using_cuda:
        torch.cuda.set_rng_state_all(rng_state["cuda"])

def get_perplexity(loss:float)->float:
    return math.exp(min(loss, 20)) # avoid overflow

def get_peak_memory(device:str):
    if torch.device(device).type != "cuda":
        return 0

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return peak_memory

def train(train_data:np.ndarray,
          eval_data:np.ndarray,
          batch_size:int,
          context_length:int,
          max_iter:int,
          lr_max:float, 
          lr_min:float, 
          T_warmup:int, 
          T_cos:int,
          device:str,
          dtype:torch.dtype,
          optimizer:torch.optim.Optimizer,
          model:torch.nn.Module,
          loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          checkpoint_interval:int,
          eval_interval:int,
          num_eval_batch:int,
          run_dir:str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
          grad_acc_step:int=1,
          max_norm:float | None = None,
          start_step:int=0,
          s_lab:swanlab.Run | None = None
          ):
    model.train()
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    is_resume = True if start_step > 0 else False
    logger = Logger(os.path.join(run_dir, "log.txt"), s_lab, is_resume)

    os.makedirs(checkpoint_dir, exist_ok=True)

    if start_step > 0:
        logger.log_info(f"resume from step {start_step}")

    for step in range(start_step+1, max_iter + 1):
        lr = lr_cos_schedule(step, lr_max, lr_min, T_warmup, T_cos)
        for params_group in optimizer.param_groups:
            params_group["lr"] = lr

        loss_acc = 0.0 #累计loss
        t0 = time.time()


        for _ in range(grad_acc_step):
            x, y = get_batch(train_data, batch_size, context_length, device)

            with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
                logits = model(x)
            
                loss = loss_fn(logits, y)
                loss /= grad_acc_step

            loss.backward()
            loss_acc += loss.detach().item()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = batch_size * context_length * grad_acc_step / dt

        perplexity = get_perplexity(loss_acc)
        peak_memory = get_peak_memory(device=device)
        # metrics
        metrics = {
            "step":f"{step}/{max_iter}",
            "train_loss":f"{loss_acc:.4f}",
            "train_perplexity":f"{perplexity:.4f}",
            "train_lr":f"{lr:.4e}",
            "train_tokens/sec":f"{tokens_per_sec:.2f}",
            "train_memory":f"{peak_memory:.1f}MB"
        }

        swanlab_metrics = {
            "train/loss": loss_acc,
            "train/perplexity": perplexity,
            "train/lr": lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/peak_memory": peak_memory,
            "step": step,
        }
        if max_norm is not None:
            norm = gradient_clip(model.parameters(), max_norm)
            metrics["train_grad_norm"] = norm
            swanlab_metrics["train/grad_norm"] = norm

        logger.log_info(metrics)
        logger.log_swanlab(swanlab_metrics)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

       
        

        if step % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
            s_lab_id = s_lab.id if s_lab else None
            save_checkpoint(model, optimizer, step, checkpoint_path, s_lab_id)
            logger.log_info("write checkpoint to " + checkpoint_path)

        if step % eval_interval == 0 or step == max_iter:
            evaluate(eval_data=eval_data,
                     curr_step=step,
                     max_step=max_iter,
                     batch_size=batch_size,
                     context_length=context_length,
                     num_eval_batch=num_eval_batch,
                     device=device,
                     dtype=dtype,
                     loss_fn=loss_fn,
                     model=model,
                     logger=logger)
            
    swanlab.finish()
        
def evaluate(eval_data:np.ndarray,
             curr_step:int,
             max_step:int, 
             batch_size:int, 
             context_length:int,
             num_eval_batch:int, 
             device:str, 
             dtype:torch.dtype, 
             loss_fn:Callable, 
             model:torch.nn.Module,
             logger:Logger):

    # 保护rng状态
    rng_state = {
        "torch":torch.get_rng_state(),
        "numpy":np.random.get_state(),
        "python":random.getstate()
    }
    if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

    # 记录初始状态
    is_training = model.training

    try:
        model.eval()
        if curr_step == max_step:
            num_eval_batch *= 3

        with torch.no_grad():
            eval_loss = 0.0
            for _ in range(num_eval_batch):
                x, y = get_batch(eval_data, batch_size, context_length, device)
                with torch.autocast(torch.device(device).type, dtype=dtype):
                    logits = model(x)
                    eval_loss += loss_fn(logits, y).item()

            eval_loss /= num_eval_batch

        metrics = {
            "step":f"{curr_step}/{max_step}",
            "eval_loss":f"{eval_loss:.4f}",
            "eval_perplexity":f"{get_perplexity(eval_loss):.4f}"
        }

        swanlab_metrics = {
            "eval/loss": eval_loss,
            "eval/perplexity": get_perplexity(eval_loss),
            "eval/peak_memory": get_peak_memory(device),
            "step": curr_step,
        }
        logger.log_info(metrics)
        logger.log_swanlab(swanlab_metrics)
    finally:
        model.train(is_training)
        using_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
        setup_rng_state(rng_state, using_cuda)

            

if __name__ == "__main__":
    # 1.argparse读入超参数
    parser = argparse.ArgumentParser(description="cs336 assigment1")
    parser.add_argument("--config", type=str, help="config file path, include")
    
    parser.add_argument("--resume_from", type=str, help="config file")
    args = parser.parse_args()

    # 2.提取超参数
    config = load_config(args.config)
    train_data_path = config.train_data
    eval_data_path = config.eval_data
    device = config.device
    dtype = getattr(torch, config.dtype.split(".")[-1])
    batch_size = config.batch_size
    context_length = config.context_length
    max_iter = config.max_iter
    lr_max = config.lr_max
    lr_min = config.lr_min
    T_warmup = config.T_warmup
    T_cos = config.T_cos
    grad_acc_step = config.grad_acc_step
    max_norm = config.max_norm
    checkpoint_interval = config.checkpoint_interval
    eval_interval = config.eval_interval
    num_eval_batch = config.num_eval_batch
    run_dir = config.run_dir
    os.makedirs(run_dir, exist_ok=True)
    seed = config.seed
    using_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    use_swanlab = config.use_swanlab
    swanlab_project_name = config.swanlab_project_name if use_swanlab else None

    # 3.准备数据
    train_data = load_data(train_data_path)
    eval_data = load_data(eval_data_path)

    # 4.设置随机数，准备模型
    
    setup_seed(seed, using_cuda)
    model = Transformer(config.model, device=torch.device(device), dtype=dtype)
    model.to(device=torch.device(device), dtype=dtype)

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    not_decay_params = [p for p in model.parameters() if p.dim() < 2]
    params_group = [{"params":decay_params}, {"params":not_decay_params, "weight_decay":0.0}]

    # 5.准备优化器
    optimizer = AdamW(params_group, **config.optimizer)
    # 6.training loop
    start_step = 0
    s_lab = None
    swanlab_id = None
    resume = None
    if args.resume_from:
        start_step, rng_state, swanlab_id = load_checkpoint(args.resume_from, model, optimizer)
        setup_rng_state(rng_state,using_cuda)
        resume = "must"

    if swanlab_project_name is not None:
        swanlab_config = {
            "batch_size":batch_size,
            "context_length":context_length,
            "max_iter":max_iter,
            "lr_max":lr_max,
            "lr_min":lr_min,
            "T_warmup":T_warmup,
            "T_cos":T_cos,
            "grad_acc_step":grad_acc_step,
            "max_norm":max_norm,
            "seed":seed,
            "device":device,
            "dtype":str(dtype),
            "model":dict(config.model),
            "optimizer":dict(config.optimizer)
        }
        s_lab = swanlab.init(project=swanlab_project_name, id=swanlab_id, resume=resume, config=swanlab_config)


    train(train_data=train_data,
          eval_data=eval_data,
          batch_size=batch_size,
          context_length=context_length,
          max_iter=max_iter,
          lr_max=lr_max,
          lr_min=lr_min,
          T_warmup=T_warmup,
          T_cos=T_cos,
          device=device,
          dtype=dtype,
          optimizer=optimizer,
          model=model,
          loss_fn=cross_entropy,
          checkpoint_interval=checkpoint_interval,
          eval_interval=eval_interval,
          num_eval_batch=num_eval_batch,
          run_dir=run_dir,
          grad_acc_step=grad_acc_step,
          max_norm=max_norm,
          start_step=start_step,
          s_lab=s_lab)
    


   