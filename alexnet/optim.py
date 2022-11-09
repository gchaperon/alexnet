import collections
import torch
import typing as tp


class _DefaultsT(tp.TypedDict):
    lr: float
    momentum: float
    weight_decay: float


class _ParamStateT(tp.TypedDict):
    running_momentum: torch.Tensor


def _default_state_factory() -> _ParamStateT:
    return dict(running_momentum=torch.tensor(0.0))


_StateT = tp.Mapping[torch.Tensor, _ParamStateT]


class _ParamGroupT(_DefaultsT):
    params: tp.List[torch.Tensor]
    # and all the other keys from _DefaultsT


class AlexNetSGD(torch.optim.Optimizer):
    """In the end I didn't use this because there was no perfoamnce improvement
    compared to torch.optim.SGD. Still, it was interesting writing my own
    optimizer"""

    defaults: _DefaultsT  # type:ignore[assignment]
    state: _StateT  # type:ignore[assignment]
    param_groups: tp.List[_ParamGroupT]  # type:ignore[assignment]

    def __init__(
        self,
        params: tp.Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        # reveal_type(self.state)  # noqa: F81
        assert lr >= 0.0, f"invalid {lr=}"
        assert momentum >= 0.0, f"invalid {momentum=}"
        assert weight_decay >= 0.0, f"invalid {weight_decay=}"

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.state = collections.defaultdict(_default_state_factory)

    @torch.no_grad()
    def step(
        self, closure: tp.Optional[tp.Callable[[], torch.Tensor]] = None
    ) -> tp.Optional[torch.Tensor]:
        # This step is just copied from the official torch SGD impl, to keep my
        # interface somewhat consistent
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # actual SGD step
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise ValueError("Found param with sparse grad")

                # assert here only for type narrowing
                assert isinstance(param.grad, torch.Tensor)
                # NOTE: See section 5 in the paper
                # Also, the first value of current_momentum is a 0-dim tensor
                # (see _default_state_factory) so this relies on shape
                # broadcasting
                current_momentum = self.state[param]["running_momentum"]

                next_momentum = (
                    group["momentum"] * current_momentum
                    - group["weight_decay"] * group["lr"] * param
                    - group["lr"] * param.grad
                )
                # NOTE: in-place add, important detail
                param += next_momentum
                self.state[param]["running_momentum"] = next_momentum

        return loss
