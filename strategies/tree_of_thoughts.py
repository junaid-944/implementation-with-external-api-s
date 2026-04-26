"""
Tree of Thoughts (ToT) Strategy.
"""
import re
import time
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from strategies.base import BaseStrategy, StrategyResult, extract_text
import config


class TaskChecker(ToTChecker):
    def evaluate(self, problem_description: str, thoughts: tuple = ()) -> ThoughtValidity:
        if not thoughts:
            return ThoughtValidity.INVALID
        last_thought = str(thoughts[-1]) if thoughts[-1] else ""
        if "final answer" in last_thought.lower() or "the answer is" in last_thought.lower():
            return ThoughtValidity.VALID_FINAL
        if len(last_thought.strip()) > 20:
            return ThoughtValidity.VALID_INTERMEDIATE
        return ThoughtValidity.INVALID


class TreeOfThoughtsStrategy(BaseStrategy):
    name = "Tree-of-Thoughts"

    TOT_PROMPT = PromptTemplate(
        input_variables=["problem_description", "thoughts"],
        template=(
            "Task: {problem_description}\n\n"
            "Previous reasoning steps:\n{thoughts}\n\n"
            "Generate the next best reasoning step only. Mention the tool needed, if any, and keep the plan short. "
            "If the final answer is known, write 'Final Answer: <answer>'."
        ),
    )

    def run(self, task: dict) -> StrategyResult:
        start_time = time.time()
        trace = []
        tool_calls = 0
        step_count = 0
        final_answer = ""
        error_message = None

        try:
            checker = TaskChecker()
            tot_chain = ToTChain(
                llm=self.llm,
                checker=checker,
                k=config.TOT_MAX_DEPTH,
                c=config.TOT_BRANCHES,
                verbose=True,
            )

            step_count += 1
            tot_result = self._invoke_with_retry(tot_chain, {"problem_description": task["description"]})
            tot_output = extract_text(tot_result.get("response", ""))
            trace.append(("tot_reasoning", tot_output[:500]))

            final_match = re.search(r"(?:Final Answer|The answer is)[:\s]*(.+?)(?:\n|$)", tot_output, re.IGNORECASE)
            if final_match:
                trace.append(("tot_proposed_answer", final_match.group(1).strip()))

            execution_prompt = (
                f"Original task: {task['description']}\n\n"
                f"Plan:\n{tot_output}\n\n"
                "Execute the shortest valid tool sequence. Prefer wikipedia_lookup for stable facts and calculator for math. "
                "Stop as soon as the required final answer is known."
            )

            agent = create_react_agent(
                self.llm,
                tools=self.tools,
                prompt="You are an execution agent. Follow the plan, avoid redundant searches, and return only the requested final answer.",
            )

            step_count += 1
            exec_result = self._invoke_with_retry(agent, {"messages": [HumanMessage(content=execution_prompt)]})
            messages = exec_result.get("messages", [])
            for msg in messages:
                msg_type = type(msg).__name__
                content = extract_text(getattr(msg, "content", ""))
                if msg_type == "AIMessage":
                    step_count += 1
                    trace.append(("execution_step", content[:300]))
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls += len(msg.tool_calls)
                        for tc in msg.tool_calls:
                            trace.append(("tool_call", f"{tc.get('name', '?')}({tc.get('args', {})})"))
                elif msg_type == "ToolMessage":
                    trace.append(("tool_result", content[:300]))

            for msg in reversed(messages):
                if type(msg).__name__ == "AIMessage":
                    text = extract_text(msg.content)
                    if text and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        final_answer = text.strip()
                        break

            if not final_answer:
                for msg in reversed(messages):
                    text = extract_text(getattr(msg, "content", ""))
                    if text:
                        final_answer = text.strip()
                        break

            final_answer = self.clean_final_answer(final_answer, task)
            trace.append(("final_answer", final_answer[:300] if final_answer else ""))

        except Exception as e:
            error_message = str(e)
            trace.append(("error", error_message))

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
            step_count=step_count,
            reasoning_trace=trace,
            tool_calls=tool_calls,
            error_message=error_message,
            metadata={"branches": config.TOT_BRANCHES, "max_depth": config.TOT_MAX_DEPTH},
        )
        result.failure_type = self.classify_failure(result, trace)
        return result

    def _invoke_with_retry(self, runnable, input_data, max_retries=None):
        retries = max_retries or config.MAX_RETRIES
        for attempt in range(retries + 1):
            try:
                try:
                    return runnable.invoke(input_data, config={"recursion_limit": config.AGENT_RECURSION_LIMIT})
                except TypeError:
                    return runnable.invoke(input_data)
            except Exception as e:
                err = str(e).lower()
                is_retryable = "429" in err or "resource_exhausted" in err or "503" in err or "unavailable" in err
                if is_retryable and attempt < retries:
                    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"              [Retry {attempt+1}/{retries}] Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                else:
                    raise
