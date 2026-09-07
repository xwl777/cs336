import argparse
import torch
import os
import swanlab

from cs336_basics.train import *
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
    swanlab_name = config.swanlab_name if use_swanlab else None

    # 3.准备数据
    train_data = load_data(train_data_path)
    eval_data = load_data(eval_data_path)

    # 4.设置随机数，准备模型
    
    setup_seed(seed, using_cuda)
    model = Transformer(**config.model, device=torch.device(device), dtype=dtype)
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
        s_lab = swanlab.init(project=swanlab_project_name, id=swanlab_id, resume=resume, config=swanlab_config, name=swanlab_name)


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