#!/usr/bin/env python3
"""Plot GRPO vs SDAR WebShop validation curves from training logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(
    r"step:(?P<step>\d+) - .*?"
    r"val/text/test_score:(?P<test>[0-9.]+).*?"
    r"val/success_rate:(?P<success>[0-9.]+).*?"
    r"val/webshop_task_score \(not success_rate\):(?P<webshop>[0-9.]+)"
)


def extract_metrics(paths: list[Path]) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for path in paths:
        text = ANSI_RE.sub("", path.read_text(errors="replace"))
        for match in METRIC_RE.finditer(text):
            step = int(match.group("step"))
            by_step[step] = {
                "step": step,
                "test_score": float(match.group("test")),
                "success_rate": float(match.group("success")),
                "webshop_score": float(match.group("webshop")),
            }
    return [by_step[step] for step in sorted(by_step)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-initial-log", type=Path, required=True)
    parser.add_argument("--grpo-resume-log", type=Path, required=True)
    parser.add_argument("--sdar-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # The resumed GRPO log contains a fresh stochastic evaluation at step 5.
    # Keep only step 0 from the original log and use the continuous resumed run
    # for steps 5--150, avoiding two different values at the same step.
    grpo_initial = [
        row for row in extract_metrics([args.grpo_initial_log]) if row["step"] == 0
    ]
    grpo = grpo_initial + extract_metrics([args.grpo_resume_log])
    sdar = extract_metrics([args.sdar_log])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "validation_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "step", "success_rate", "webshop_score", "test_score"],
        )
        writer.writeheader()
        for method, rows in (("GRPO", grpo), ("SDAR", sdar)):
            for row in rows:
                writer.writerow({"method": method, **row})

    colors = {"GRPO": "#4472C4", "SDAR": "#E15759"}
    datasets = {"GRPO": grpo, "SDAR": sdar}
    panels = [
        ("success_rate", "Validation Success Rate", lambda value: value * 100, "%"),
        ("webshop_score", "WebShop Task Score", lambda value: value, ""),
        ("test_score", "Validation Test Score", lambda value: value, ""),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (key, title, transform, suffix) in zip(axes, panels):
        for method, rows in datasets.items():
            x = [int(row["step"]) for row in rows]
            y = [transform(float(row[key])) for row in rows]
            ax.plot(
                x,
                y,
                color=colors[method],
                marker="o",
                markersize=3.5,
                linewidth=2,
                label=method,
            )
            best_index = max(range(len(y)), key=y.__getitem__)
            ax.scatter(x[best_index], y[best_index], color=colors[method], s=55, zorder=4)
            ax.annotate(
                f"best {y[best_index]:.1f}{suffix}" if suffix else f"best {y[best_index]:.3f}",
                (x[best_index], y[best_index]),
                xytext=(4, 7),
                textcoords="offset points",
                fontsize=8,
                color=colors[method],
            )
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 150)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Success rate (%)")
    axes[1].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    fig.suptitle("Qwen3-1.7B on WebShop-small: GRPO vs SDAR", fontsize=15)

    png_path = args.output_dir / "validation_curves.png"
    pdf_path = args.output_dir / "validation_curves.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
