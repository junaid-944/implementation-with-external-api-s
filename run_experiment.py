"""
Main experiment runner for the Agentic Planning Study.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import config
from strategies import ReActStrategy, PlanAndSolveStrategy, TreeOfThoughtsStrategy
from tasks import get_tasks_by_complexity
from evaluation.metrics import compute_metrics, compute_metrics_by_complexity, compute_failure_analysis
from evaluation.visualizer import ResultVisualizer

STRATEGIES = {
    "ReAct": ReActStrategy,
    "Plan-and-Solve": PlanAndSolveStrategy,
    "Tree-of-Thoughts": TreeOfThoughtsStrategy,
}


def run_single_task(strategy, task: dict, task_num: int, total: int):
    print(f"    [{task_num}/{total}] Task: {task['id']} ({task['complexity']})")
    print(f"           {task['description'][:80]}...")
    result = strategy.run(task)
    status = "PASS" if result.correct else "FAIL"
    answer_preview = result.final_answer[:60] if result.final_answer else "(no answer)"
    print(f"           -> {status} | Answer: {answer_preview}")
    print(f"              Time: {result.execution_time:.1f}s | Steps: {result.step_count} | Tools: {result.tool_calls}")
    if result.failure_type:
        print(f"              Failure: {result.failure_type}")
    if result.error_message:
        print(f"              Error: {result.error_message[:120]}")
    print()
    return result


def select_tasks(quick: bool):
    tasks = []
    per_complexity = 1 if quick else 2
    for complexity in ["low", "medium", "high"]:
        tasks.extend(get_tasks_by_complexity(complexity)[:per_complexity])
    return tasks


def run_experiment(strategy_filter: str = None, quick: bool = False):
    print("=" * 70)
    print("  AGENTIC PLANNING STUDY — EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Model: {config.OPENAI_MODEL}")
    print(f"  Temperature: {config.TEMPERATURE}")
    print(f"  Max steps/task: {config.MAX_STEPS_PER_TASK}")
    print(f"  Max tool calls/task: {config.MAX_TOOL_CALLS_PER_TASK}")
    print(f"  Runs per task: {config.NUM_RUNS_PER_TASK}")
    print(f"  Mode: {'Quick test' if quick else 'Full experiment'}")
    print("=" * 70)

    tasks = select_tasks(quick)
    num_runs = 1 if quick else config.NUM_RUNS_PER_TASK

    if strategy_filter:
        if strategy_filter not in STRATEGIES:
            print(f"Unknown strategy: {strategy_filter}")
            print(f"Available: {', '.join(STRATEGIES.keys())}")
            return
        strategies_to_run = {strategy_filter: STRATEGIES[strategy_filter]}
    else:
        strategies_to_run = STRATEGIES

    print(f"\n  Tasks: {len(tasks)}")
    print(f"  Strategies: {', '.join(strategies_to_run.keys())}")
    print(f"  Total runs: {len(tasks) * len(strategies_to_run) * num_runs}")
    print()

    all_results = []
    all_run_results = {}

    for run_num in range(1, num_runs + 1):
        print(f"\n{'='*70}")
        print(f"  RUN {run_num}/{num_runs}")
        print(f"{'='*70}")
        run_results = []
        for strategy_name, strategy_class in strategies_to_run.items():
            print(f"\n  --- Strategy: {strategy_name} ---\n")
            strategy = strategy_class()
            for i, task in enumerate(tasks):
                result = run_single_task(strategy, task, i + 1, len(tasks))
                all_results.append(result)
                run_results.append(result)
        all_run_results[run_num] = run_results

    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    raw_results_path = os.path.join(config.RESULTS_DIR, "raw_results.json")
    raw_data = [r.to_dict() for r in all_results]
    with open(raw_results_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"  Raw results: {raw_results_path}")

    print("\n  Generating visualizations and metrics...")
    viz = ResultVisualizer(all_results, all_run_results)
    viz.generate_all()

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    metrics = compute_metrics(all_results)
    if not metrics.empty:
        for _, row in metrics.iterrows():
            print(f"\n  {row['strategy']}:")
            print(f"    Success Rate:     {row['success_rate']:.1f}%")
            print(f"    Avg Time:         {row['avg_execution_time']:.2f}s")
            print(f"    Avg Steps:        {row['avg_step_count']:.1f}")
            print(f"    Total Tool Calls: {row['total_tool_calls']}")

    complexity_metrics = compute_metrics_by_complexity(all_results)
    if not complexity_metrics.empty:
        print("\n  --- By Complexity ---")
        for _, row in complexity_metrics.iterrows():
            print(f"    {row['strategy']} | {row['complexity']:>6s}: {row['success_rate']:.1f}% success")

    failures = compute_failure_analysis(all_results)
    if not failures.empty:
        print("\n  --- Failure Analysis ---")
        for _, row in failures.iterrows():
            print(f"    {row['strategy']}: {row['failure_type']} ({row['count']})")

    print(f"\n  All results saved to: {config.RESULTS_DIR}/")
    print("=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentic Planning Study experiment")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer tasks, 1 run)")
    parser.add_argument("--strategy", type=str, default=None, help="Run only one strategy (ReAct, Plan-and-Solve, Tree-of-Thoughts)")
    args = parser.parse_args()
    run_experiment(strategy_filter=args.strategy, quick=args.quick)
