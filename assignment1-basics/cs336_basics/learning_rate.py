import math
def lr_cos_schedule(t:int, lr_max:float, lr_min:float, T_warmup:int, T_cos:int):
    if t < T_warmup:
        return lr_max * t / T_warmup
    elif t < T_cos:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos((t - T_warmup) / (T_cos - T_warmup) * math.pi))
    else:
        return lr_min