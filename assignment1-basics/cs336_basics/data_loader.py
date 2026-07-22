import numpy as np
import torch
def get_batch(x:np.ndarray, batch_size:int, context_length:int, device:str):
    max_start_idx = len(x) - context_length - 1

    start_idxs = np.random.randint(0, max_start_idx + 1, size=batch_size)

    x_seq = np.zeros((batch_size, context_length), dtype=np.int64)
    y_seq = np.zeros((batch_size, context_length), dtype=np.int64)

    for i, idx in enumerate(start_idxs):
        x_seq[i] = x[idx:idx + context_length]
        y_seq[i] = x[idx + 1:idx+ context_length + 1]

    x_batch = torch.from_numpy(x_seq)
    y_batch = torch.from_numpy(y_seq)

    if device.startswith("cuda"):
        x_batch, y_batch = (
            x_batch.pin_memory().to(device, non_blocking=True),
            y_batch.pin_memory().to(device, non_blocking=True),
        )
    else:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    return x_batch, y_batch
