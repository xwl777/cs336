import torch
import os
import typing
import numpy as np
import random
def save_checkpoint(model:torch.nn.Module, 
                    optimizer:torch.optim.Optimizer, 
                    iteration:int, 
                    out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    swanlab_id:str | None = None):
    rng_state = {
        "torch":torch.get_rng_state(),
        "numpy":np.random.get_state(),
        "python":random.getstate()
    }
    if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
    message = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration,
        "rng_state":rng_state,
        "swanlab_id":swanlab_id
    }
    torch.save(message, out)

def load_checkpoint(src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model:torch.nn.Module, 
                    optimizer:torch.optim.Optimizer):
    message = torch.load(src)
    model.load_state_dict(message["model"])
    optimizer.load_state_dict(message["optimizer"])
    return message["iteration"], message["rng_state"],message["swanlab_id"]
