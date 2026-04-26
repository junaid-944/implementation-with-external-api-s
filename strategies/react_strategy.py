"""
ReAct Strategy: Synergizing Reasoning and Acting.
"""
import time
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from strategies.base import BaseStrategy, StrategyResult, extract_text
import config


class ReActStrategy(BaseStrategy):
    name = "ReAct"

    SYSTEM_PROMPT = (
        "You are an AI agent that solves tasks using the ReAct framework. "
        "Use tools only when needed. Stop as soon as the task is solved.\n\n"
        "Rules:\n"
        "- Prefer calculator first for pure arithmetic.\n"
        "- Prefer wikipedia_lookup before web_search for stable factual lookups.\n"
        "- Avoid repeating the same search with only tiny wording changes.\n"
        "- Final answer must be concise and contain only the information requested.\n"
    )

    def run(self, task: dict) -> StrategyResult:
        start_time = time.time()
        trace = []
        tool_calls = 0
        step_count = 0
        final_answer = ""
        error_message = None

        try:
            agent = create_react_agent(
                self.llm,
                tools=self.tools,
                prompt=self.SYSTEM_PROMPT,
            )

            result_state = self._invoke_with_retry(
                agent,
                {"messages": [HumanMessage(content=f"Task: {task['description']}")]},
            )

            messages = result_state.get("messages", [])
            for msg in messages:
                msg_type = type(msg).__name__
                content = extract_text(getattr(msg, "content", ""))

                if msg_type == "HumanMessage":
                    trace.append(("input", content[:200]))
                elif msg_type == "AIMessage":
                    step_count += 1
                    trace.append(("reasoning", content[:300]))
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
            trace.append(("final_answer", final_answer[:300]))

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
        )
        result.failure_type = self.classify_failure(result, trace)
        return result

    def _invoke_with_retry(self, agent, input_data, max_retries=None):
        retries = max_retries or config.MAX_RETRIES
        for attempt in range(retries + 1):
            try:
                return agent.invoke(input_data, config={"recursion_limit": config.AGENT_RECURSION_LIMIT})
            except Exception as e:
                err = str(e).lower()
                is_retryable = "429" in err or "resource_exhausted" in err or "503" in err or "unavailable" in err
                if is_retryable and attempt < retries:
                    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"              [Retry {attempt+1}/{retries}] Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                else:
                    raise
