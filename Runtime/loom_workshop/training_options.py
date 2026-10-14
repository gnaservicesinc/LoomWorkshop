"""Optimizer and sampling policy for the current, fixed-base Workshop pipeline."""
import hashlib
import math
import random

import torch


class Lion(torch.optim.Optimizer):
    """Sign updates with decoupled decay and a separate momentum EMA."""
    def __init__(self, parameters, lr, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(parameters, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise ValueError("Lion requires dense gradients.")
                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                average = state["exp_avg"]
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                direction = average.mul(beta1).add(gradient, alpha=1 - beta1).sign_()
                parameter.add_(direction, alpha=-group["lr"])
                average.mul_(beta2).add_(gradient, alpha=1 - beta2)
        return loss


def validate_options(request):
    options = dict(optimizer="adamw", gradient_accumulation_steps=1, gradient_clip_norm=1.0,
                   weight_decay=0.01, beta1=0.9,
                   beta2=0.99 if request.get("optimizer") == "lion" else 0.999, momentum=0.9,
                   lr_schedule="constant", warmup_epochs=0, minimum_lr_ratio=0.1,
                   crop_refresh="per_sample", augmentation_refresh="per_sample", seed=42,
                   checkpoint_every_epochs=1)
    options.update({key: request[key] for key in options if key in request})
    if options["optimizer"] not in ("adamw", "adam", "sgd", "lion"):
        raise ValueError("Choose AdamW, Adam, SGD, or Lion.")
    if options["optimizer"] == "sgd":
        options.update(beta1=0.9, beta2=0.999)
    else:
        options["momentum"] = 0.9
    if options["lr_schedule"] not in ("constant", "cosine"):
        raise ValueError("Choose a constant or cosine learning rate schedule.")
    for key in ("crop_refresh", "augmentation_refresh"):
        if options[key] not in ("per_sample", "per_epoch", "per_run"):
            raise ValueError(f"Unsupported {key} timing.")
    for key, low, high in (("gradient_accumulation_steps", 1, 4096), ("warmup_epochs", 0, 100000),
                           ("seed", 0, 2**31 - 1), ("checkpoint_every_epochs", 1, 100000)):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
    if options["warmup_epochs"] >= request.get("epochs", 10):
        raise ValueError("Warmup epochs must be fewer than the total training epochs.")
    for key in ("weight_decay", "gradient_clip_norm", "beta1", "beta2", "momentum", "minimum_lr_ratio"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    for key in ("beta1", "beta2", "momentum"):
        if options[key] >= 1:
            raise ValueError(f"{key} must be below 1.")
    if options["minimum_lr_ratio"] > 1:
        raise ValueError("The minimum learning rate ratio must be between 0 and 1.")
    if options["weight_decay"] > 10:
        raise ValueError("Weight decay must be between 0 and 10.")
    if options["gradient_clip_norm"] > 1000000:
        raise ValueError("Gradient clip norm must be between 0 and 1,000,000.")
    rate = request.get("learning_rate", 0.001)
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("Learning rate must be greater than 0 and at most 1.")
    for key in ("shuffle_each_epoch", "flip_crops", "rotate_crops", "exaggerate_features"):
        if key in request and not isinstance(request[key], bool):
            raise ValueError(f"{key} must be true or false.")
    return options


def make_optimizer(parameters, learning_rate, options):
    common = dict(lr=learning_rate, weight_decay=options["weight_decay"])
    if options["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, momentum=options["momentum"], **common)
    betas = (options["beta1"], options["beta2"])
    if options["optimizer"] == "lion":
        return Lion(parameters, betas=betas, **common)
    constructor = torch.optim.AdamW if options["optimizer"] == "adamw" else torch.optim.Adam
    return constructor(parameters, betas=betas, **common)


def learning_rate_at(step, total_steps, steps_per_epoch, peak, options):
    warmup = min(options["warmup_epochs"] * steps_per_epoch, total_steps)
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    if options["lr_schedule"] == "constant":
        return peak
    span = total_steps - warmup
    fraction = 0.0 if span <= 1 else (step - warmup) / (span - 1)
    factor = options["minimum_lr_ratio"] + (1 - options["minimum_lr_ratio"]) * (1 + math.cos(math.pi * fraction)) / 2
    return peak * factor


def sampling_rng(mode, seed, stream, sample_id, epoch, advancing_rng):
    if mode == "per_sample":
        return advancing_rng
    marker = epoch if mode == "per_epoch" else 0
    digest = hashlib.sha256(f"{seed}|{stream}|{sample_id}|{marker}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def accumulation_windows(order, batch_size, accumulation_steps):
    """Bounded windows contain the actual samples used to weight their losses."""
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    for start in range(0, len(batches), accumulation_steps):
        window = batches[start:start + accumulation_steps]
        yield window, sum(map(len, window))
