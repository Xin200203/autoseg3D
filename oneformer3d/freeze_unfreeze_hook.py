import re
from typing import Iterable, Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FreezeUnfreezeHook(Hook):
    """Freeze/unfreeze parameters by name patterns.

    This is designed for quick finetuning experiments where you introduce new
    modules (e.g. track-window STM) and want a stable warmup phase.

    Args:
        warmup_epochs: Number of initial epochs to train only `warmup_trainable`.
        warmup_trainable: Iterable of regex patterns. If any matches a parameter
            name, it is trainable during warmup.
        always_frozen: Iterable of regex patterns. If any matches a parameter
            name, it is kept frozen for all epochs.
        verbose: Whether to log trainable parameter counts.
    """

    priority = "VERY_HIGH"

    def __init__(
        self,
        warmup_epochs: int = 1,
        warmup_trainable: Optional[Iterable[str]] = None,
        always_frozen: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> None:
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.warmup_trainable = list(warmup_trainable or [])
        self.always_frozen = list(always_frozen or [])
        self.verbose = bool(verbose)

        self._compiled_warmup = [re.compile(p) for p in self.warmup_trainable]
        self._compiled_always = [re.compile(p) for p in self.always_frozen]

    def _is_always_frozen(self, name: str) -> bool:
        return any(p.search(name) is not None for p in self._compiled_always)

    def _is_warmup_trainable(self, name: str) -> bool:
        return any(p.search(name) is not None for p in self._compiled_warmup)

    def _apply(self, runner, phase: str) -> None:
        model = runner.model
        if hasattr(model, "module"):
            model = model.module

        n_total = 0
        n_trainable = 0
        for name, param in model.named_parameters():
            n_total += param.numel()

            trainable = True
            if self._is_always_frozen(name):
                trainable = False
            elif phase == "warmup":
                trainable = self._is_warmup_trainable(name)
            else:
                trainable = True

            param.requires_grad = bool(trainable)
            if trainable:
                n_trainable += param.numel()

        if self.verbose and hasattr(runner, "logger"):
            runner.logger.info(
                "[FreezeUnfreezeHook] phase=%s warmup_epochs=%d trainable_params=%d/%d (%.2f%%)",
                phase,
                int(self.warmup_epochs),
                int(n_trainable),
                int(n_total),
                100.0 * float(n_trainable) / float(max(n_total, 1)),
            )

    def before_train(self, runner) -> None:
        phase = "warmup" if self.warmup_epochs > 0 else "full"
        self._apply(runner, phase=phase)

    def before_train_epoch(self, runner) -> None:
        # At the first epoch after warmup ends, switch to full finetune.
        if self.warmup_epochs > 0 and int(runner.epoch) == int(self.warmup_epochs):
            self._apply(runner, phase="full")

