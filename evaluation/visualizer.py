"""
Visualization module for experiment results.
Generates charts and tables for the comparative study.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
from typing import List
from strategies.base import StrategyResult
from evaluation.metrics import (
    compute_metrics,
    compute_metrics_by_complexity,
    compute_failure_analysis,
    compute_stability_metrics,
)
import config


# Color palette
STRATEGY_COLORS = {
    "ReAct": "#2196F3",
    "Plan-and-Solve": "#4CAF50",
    "Tree-of-Thoughts": "#FF9800",
}
COMPLEXITY_ORDER = ["low", "medium", "high"]


class ResultVisualizer:
    """Generate all charts and summary tables."""

    def __init__(self, results: List[StrategyResult], all_runs: dict = None):
        self.results = results
        self.all_runs = all_runs or {}
        self.output_dir = config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.1)

    def _save(self, fig, name: str):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # -----------------------------------------------------------------
    # 1. Overall Success Rate Bar Chart
    # -----------------------------------------------------------------
    def plot_success_rate(self):
        metrics = compute_metrics(self.results)
        fig, ax = plt.subplots(figsize=(8, 5))
        strategies = metrics["strategy"].tolist()
        colors = [STRATEGY_COLORS.get(s, "#999") for s in strategies]
        bars = ax.bar(strategies, metrics["success_rate"], color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, metrics["success_rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Overall Task Success Rate by Strategy")
        ax.set_ylim(0, 105)
        self._save(fig, "01_success_rate.png")

    # -----------------------------------------------------------------
    # 2. Success Rate by Complexity (Grouped Bar Chart)
    # -----------------------------------------------------------------
    def plot_success_by_complexity(self):
        metrics = compute_metrics_by_complexity(self.results)
        if metrics.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        strategies = metrics["strategy"].unique()
        x = np.arange(len(COMPLEXITY_ORDER))
        width = 0.25

        for i, strategy in enumerate(strategies):
            subset = metrics[metrics["strategy"] == strategy]
            rates = []
            for c in COMPLEXITY_ORDER:
                row = subset[subset["complexity"] == c]
                rates.append(row["success_rate"].values[0] if len(row) > 0 else 0)
            color = STRATEGY_COLORS.get(strategy, "#999")
            bars = ax.bar(x + i * width, rates, width, label=strategy, color=color, edgecolor="white")
            for bar, val in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Task Complexity")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate by Complexity Level")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_ylim(0, 115)
        ax.legend()
        self._save(fig, "02_success_by_complexity.png")

    # -----------------------------------------------------------------
    # 3. Execution Time Comparison
    # -----------------------------------------------------------------
    def plot_execution_time(self):
        records = [r.to_dict() for r in self.results]
        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(9, 5))
        strategies = df["strategy"].unique()
        data_to_plot = [df[df["strategy"] == s]["execution_time"].values for s in strategies]
        colors = [STRATEGY_COLORS.get(s, "#999") for s in strategies]

        bp = ax.boxplot(data_to_plot, labels=strategies, patch_artist=True, widths=0.4)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("Execution Time Distribution by Strategy")
        self._save(fig, "03_execution_time.png")

    # -----------------------------------------------------------------
    # 4. Step Count Comparison
    # -----------------------------------------------------------------
    def plot_step_count(self):
        records = [r.to_dict() for r in self.results]
        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(9, 5))
        strategies = df["strategy"].unique()
        data_to_plot = [df[df["strategy"] == s]["step_count"].values for s in strategies]
        colors = [STRATEGY_COLORS.get(s, "#999") for s in strategies]

        bp = ax.boxplot(data_to_plot, labels=strategies, patch_artist=True, widths=0.4)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Step Count")
        ax.set_title("Reasoning Step Count by Strategy")
        self._save(fig, "04_step_count.png")

    # -----------------------------------------------------------------
    # 5. Execution Time by Complexity (Grouped)
    # -----------------------------------------------------------------
    def plot_time_by_complexity(self):
        metrics = compute_metrics_by_complexity(self.results)
        if metrics.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        strategies = metrics["strategy"].unique()
        x = np.arange(len(COMPLEXITY_ORDER))
        width = 0.25

        for i, strategy in enumerate(strategies):
            subset = metrics[metrics["strategy"] == strategy]
            times = []
            for c in COMPLEXITY_ORDER:
                row = subset[subset["complexity"] == c]
                times.append(row["avg_execution_time"].values[0] if len(row) > 0 else 0)
            color = STRATEGY_COLORS.get(strategy, "#999")
            ax.bar(x + i * width, times, width, label=strategy, color=color, edgecolor="white")

        ax.set_xlabel("Task Complexity")
        ax.set_ylabel("Avg Execution Time (seconds)")
        ax.set_title("Average Execution Time by Complexity Level")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.legend()
        self._save(fig, "05_time_by_complexity.png")

    # -----------------------------------------------------------------
    # 6. Failure Analysis (Stacked Bar)
    # -----------------------------------------------------------------
    def plot_failure_analysis(self):
        failures = compute_failure_analysis(self.results)
        if failures.empty:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot = failures.pivot_table(index="strategy", columns="failure_type",
                                     values="count", fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
        ax.set_ylabel("Number of Failures")
        ax.set_title("Failure Type Distribution by Strategy")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Failure Type", bbox_to_anchor=(1.05, 1))
        self._save(fig, "06_failure_analysis.png")

    # -----------------------------------------------------------------
    # 7. Radar / Spider Chart — Multi-Metric Comparison
    # -----------------------------------------------------------------
    def plot_radar_chart(self):
        metrics = compute_metrics(self.results)
        if metrics.empty:
            return

        categories = ["Success Rate", "Speed\n(inv. time)", "Efficiency\n(inv. steps)",
                       "Tool Usage", "Reliability\n(inv. failures)"]
        N = len(categories)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for _, row in metrics.iterrows():
            strategy = row["strategy"]
            # Normalize metrics to 0-1 scale
            success = row["success_rate"] / 100
            speed = 1 - min(row["avg_execution_time"] / 30, 1)  # Inverse, cap at 30s
            efficiency = 1 - min(row["avg_step_count"] / 15, 1)  # Inverse, cap at 15
            tool_use = min(row["avg_tool_calls"] / 5, 1)
            failures = [r for r in self.results if r.strategy_name == strategy and not r.correct]
            reliability = 1 - len(failures) / max(row["total_tasks"], 1)

            values = [success, speed, efficiency, tool_use, reliability]
            values += values[:1]

            color = STRATEGY_COLORS.get(strategy, "#999")
            ax.plot(angles, values, "o-", linewidth=2, label=strategy, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Multi-Metric Strategy Comparison", y=1.08, fontsize=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        self._save(fig, "07_radar_chart.png")

    # -----------------------------------------------------------------
    # 8. Summary Table
    # -----------------------------------------------------------------
    def save_summary_table(self):
        metrics = compute_metrics(self.results)
        path = os.path.join(self.output_dir, "summary_metrics.csv")
        metrics.to_csv(path, index=False)
        print(f"  Saved: {path}")

        # Also save detailed results
        records = [r.to_dict() for r in self.results]
        detail_path = os.path.join(self.output_dir, "detailed_results.csv")
        pd.DataFrame(records).to_csv(detail_path, index=False)
        print(f"  Saved: {detail_path}")

        # Complexity breakdown
        complexity_metrics = compute_metrics_by_complexity(self.results)
        complexity_path = os.path.join(self.output_dir, "metrics_by_complexity.csv")
        complexity_metrics.to_csv(complexity_path, index=False)
        print(f"  Saved: {complexity_path}")

    # -----------------------------------------------------------------
    # Generate All
    # -----------------------------------------------------------------
    def generate_all(self):
        """Generate all charts and tables."""
        print("\nGenerating visualizations...")
        self.plot_success_rate()
        self.plot_success_by_complexity()
        self.plot_execution_time()
        self.plot_step_count()
        self.plot_time_by_complexity()
        self.plot_failure_analysis()
        self.plot_radar_chart()
        self.save_summary_table()
        print(f"\nAll results saved to: {self.output_dir}/")
