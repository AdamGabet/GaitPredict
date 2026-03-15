"""Heuristics for stopping training runs early based on evaluation loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EarlyStopDecision:
    """Summary of the latest early stop check."""

    should_stop: bool
    reason: Optional[str] = None


@dataclass
class EvalLossEarlyStopper:
    """Track evaluation loss and issue stop signals when it degrades."""

    loss_increase_patience: int = 3
    high_loss_threshold: float = 0.15
    high_loss_patience: int = 2

    prev_loss: Optional[float] = None
    loss_increase_streak: int = 0
    high_loss_streak: int = 0
    stop_reason: Optional[str] = None

    def update(self, metrics: Dict[str, float]) -> EarlyStopDecision:
        """Consume the metrics dict and decide whether to stop based on eval loss."""

        eval_loss = metrics.get('eval_loss')
        if eval_loss is None:
            return EarlyStopDecision(False, None)

        # Check for consecutive loss increases
        if self.prev_loss is not None and eval_loss > self.prev_loss:
            self.loss_increase_streak += 1
        else:
            self.loss_increase_streak = 0

        if self.loss_increase_streak >= self.loss_increase_patience:
            self.stop_reason = f"loss_increasing_{eval_loss:.4f}"

        # Check for high loss threshold
        if eval_loss > self.high_loss_threshold:
            self.high_loss_streak += 1
        else:
            self.high_loss_streak = 0

        if self.high_loss_streak >= self.high_loss_patience:
            self.stop_reason = f"loss_too_high_{eval_loss:.4f}"

        self.prev_loss = eval_loss

        return EarlyStopDecision(bool(self.stop_reason), self.stop_reason)

    def should_stop(self) -> bool:
        return self.stop_reason is not None

    def reset(self) -> None:
        self.prev_loss = None
        self.loss_increase_streak = 0
        self.high_loss_streak = 0
        self.stop_reason = None