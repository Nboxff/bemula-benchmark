"""Evaluation framework: judging, control-pair runner, metrics."""

from bemula.eval.judge import Judge, LLMJudge, MockJudge, Verdict
from bemula.eval.metrics import compute_metrics
from bemula.eval.runner import ItemRecord, run_evaluation, run_instance

__all__ = [
    "Judge",
    "LLMJudge",
    "MockJudge",
    "Verdict",
    "compute_metrics",
    "ItemRecord",
    "run_evaluation",
    "run_instance",
]
