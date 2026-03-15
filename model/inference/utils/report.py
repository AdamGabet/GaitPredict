import os
from typing import Dict, Optional

import numpy as np


def _format_scalar(value) -> str:
    """Return a readable string for numeric scalar values."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _format_vector_summary(vector: Optional[np.ndarray]) -> str:
    """Describe a high-dimensional vector without dumping every element."""
    if vector is None:
        return "None"
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    if vector.size == 0:
        return "empty"
    preview = ", ".join(f"{elem:.4f}" for elem in vector.ravel()[:5])
    if vector.size > 5:
        preview += ", ..."
    return f"len={vector.size}, first=[{preview}]"


def write_markdown_report(
    metrics: Dict[str, float],
    stats: Dict[str, Dict],
    destination_path: str,
    run_id: str,
    config: Optional[Dict] = None,
) -> str:
    """Create a Markdown summary that explains metrics and embedding stats."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    lines = []
    lines.append(f"# Evaluation Summary for `{run_id}`")
    lines.append("")

    if config:
        lines.append("## Configuration Highlights")
        if "labels" in config:
            labels_str = ", ".join(config["labels"]) if config["labels"] else "None"
            lines.append(f"- Labels: {labels_str}")
        if "task_types" in config:
            task_str = ", ".join(config["task_types"]) if config["task_types"] else "None"
            lines.append(f"- Task Types: {task_str}")
        if "size_seq" in config:
            lines.append(f"- Sequence Length: {config['size_seq']}")
        lines.append("")

    if metrics:
        lines.append("## Metrics Overview")
        lines.append("| Split | Metric | Value |")
        lines.append("|-------|--------|-------|")
        for key in sorted(metrics):
            value = _format_scalar(metrics[key])
            if "/" in key:
                split, metric_name = key.split("/", 1)
            else:
                split, metric_name = "global", key
            lines.append(f"| {split} | {metric_name} | {value} |")
        lines.append("")
    else:
        lines.append("## Metrics Overview")
        lines.append("No metrics were computed for this run.")
        lines.append("")

    lines.append("## Embedding Statistics")
    if not stats:
        lines.append("No embedding statistics were captured.")
    else:
        for split_name in sorted(stats):
            split_stats = stats[split_name]
            lines.append(f"### {split_name.capitalize()} Split")
            mean_norm = _format_scalar(split_stats.get("mean_norm"))
            trace_value = _format_scalar(split_stats.get("trace"))
            embedding_dim = (
                len(split_stats["mean"]) if split_stats.get("mean") is not None else "unknown"
            )
            lines.append(f"- Embedding Dimension: {embedding_dim}")
            lines.append(f"- Mean Vector Norm: {mean_norm}")
            lines.append(f"- Variance Trace: {trace_value}")
            lines.append(f"- Mean Preview: {_format_vector_summary(split_stats.get('mean'))}")
            lines.append(f"- Variance Preview: {_format_vector_summary(split_stats.get('var'))}")

            per_activity = split_stats.get("per_activity") or {}
            if per_activity:
                lines.append("")
                lines.append("| Activity | Samples | Mean Norm | Variance Trace |")
                lines.append("|----------|---------|-----------|----------------|")
                for activity_name in sorted(per_activity):
                    activity_stats = per_activity[activity_name]
                    count = _format_scalar(activity_stats.get("count"))
                    act_mean_norm = _format_scalar(activity_stats.get("mean_norm"))
                    act_trace = _format_scalar(activity_stats.get("trace"))
                    lines.append(
                        f"| {activity_name} | {count} | {act_mean_norm} | {act_trace} |"
                    )
                lines.append("")

            lines.append("")

    content = "\n".join(lines)
    with open(destination_path, "w", encoding="utf-8") as handle:
        handle.write(content)

    return destination_path
