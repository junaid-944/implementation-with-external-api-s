"""
Plan-and-Solve Strategy.
"""
import time
from langchain_experimental.plan_and_execute import PlanAndExecute, load_chat_planner, load_agent_executor
from langchain_core.callbacks import BaseCallbackHandler

from strategies.base import BaseStrategy, StrategyResult, extract_text
import config


class StepTracker(BaseCallbackHandler):
    def __init__(self):
        self.trace = []
        self.tool_calls = 0
        self.step_count = 0

    def on_agent_action(self, action, **kwargs):
        self.tool_calls += 1
        self.step_count += 1
        self.trace.append(("tool_call", f"{action.tool}({action.tool_input})"))

    def on_tool_end(self, output, **kwargs):
        self.trace.append(("tool_result", str(output)[:300]))

    def on_llm_end(self, response, **kwargs):
        self.step_count += 1
        try:
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                text = extract_text(getattr(gen, "text", getattr(gen, "content", "")))
                self.trace.append(("llm_output", text[:300]))
            else:
                self.trace.append(("llm_output", "(empty response)"))
        except (IndexError, AttributeError):
            self.trace.append(("llm_output", "(could not parse response)"))

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name", "unknown") if serialized else "unknown"
        if "step" in name.lower() or "execute" in name.lower():
            self.trace.append(("step_start", str(inputs)[:200]))


class PlanAndSolveStrategy(BaseStrategy):
    name = "Plan-and-Solve"

    PLANNER_PROMPT = (
        "Devise the minimum number of steps needed to solve the task accurately. "
        "Use calculator for arithmetic, wikipedia_lookup for stable facts, and web_search only when needed. "
        "Avoid redundant verification. The final step should answer the original question in the shortest correct form."
    )

    def run(self, task: dict) -> StrategyResult:
        start_time = time.time()
        final_answer = ""
        error_message = None
        tracker = StepTracker()

        for attempt in range(config.MAX_RETRIES + 1):
            try:
                planner = load_chat_planner(self.llm, system_prompt=self.PLANNER_PROMPT)
                executor = load_agent_executor(self.llm, tools=self.tools, verbose=True)
                agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
                result = agent.invoke({"input": task["description"]}, config={"callbacks": [tracker]})
                raw_output = result.get("output", "")
                final_answer = self.clean_final_answer(extract_text(raw_output).strip(), task)
                tracker.trace.append(("final_answer", final_answer))
                break
            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                is_retryable = "429" in err_lower or "resource_exhausted" in err_lower or "503" in err_lower or "unavailable" in err_lower
                if is_retryable and attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"              [Retry {attempt+1}/{config.MAX_RETRIES}] Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    continue
                error_message = err_str
                tracker.trace.append(("error", error_message))
                break

        execution_time = time.time() - start_time
        correct = self.check_answer(final_answer, task.get("expected_answer", ""))

        result = StrategyResult(
            strategy_name=self.name,
            task_id=task["id"],
            task_description=task["description"],
            final_answer=final_answer,
            correct=correct,
            expected_answer=task.get("expected_answer", ""),
            execution_time=execution_time,
            step_count=tracker.step_count,
            reasoning_trace=tracker.trace,
            tool_calls=tracker.tool_calls,
            error_message=error_message,
        )
        result.failure_type = self.classify_failure(result, tracker.trace)
        return result
