from torch import nn


def freeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Module) and hasattr(m, "reset_parameters"):
        m.reset_parameters()
