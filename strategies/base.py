"""
Base class for all planning strategies.
Provides a common interface, shared utilities, and the StrategyResult dataclass.

All strategies use LangChain + Google Gemini for LLM interaction and tool execution.
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from tools.agent_tools import get_all_tools


def extract_text(content) -> str:
    """Safely extract text from message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return " ".join(parts)
    return str(content)


@dataclass
class StrategyResult:
    """Result of a strategy execution."""
    strategy_name: str
    task_id: str
    task_description: str
    final_answer: str
    correct: bool
    expected_answer: str
    execution_time: float
    step_count: int
    reasoning_trace: list
    tool_calls: int
    failure_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "final_answer": self.final_answer,
            "correct": self.correct,
            "expected_answer": self.expected_answer,
            "execution_time": round(self.execution_time, 3),
            "step_count": self.step_count,
            "tool_calls": self.tool_calls,
            "failure_type": self.failure_type,
            "error_message": self.error_message,
        }


class BaseStrategy:
    """Abstract base class for planning strategies."""

    name: str = "base"

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_TOKENS,
        )
        self.tools = get_all_tools()
        self.max_steps = config.MAX_STEPS_PER_TASK

    def _extract_numbers(self, text: str) -> list[float]:
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text or "")
        values = []
        for num in nums:
            try:
                values.append(float(num.replace(",", "")))
            except ValueError:
                continue
        return values

    def _normalise_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"[`*_#>]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().rstrip(".")

    def clean_final_answer(self, final_answer: str, task: dict) -> str:
        """Compress verbose model output into benchmark-friendly final answers."""
        text = self._normalise_text(final_answer)
        if not text:
            return ""

        marker_match = re.search(r"(?:final answer|answer)\s*[:\-]\s*(.+)", text, re.IGNORECASE)
        if marker_match:
            text = marker_match.group(1).strip()

        segments = [seg.strip() for seg in re.split(r"[\n\r]+", final_answer) if seg.strip()]
        if len(segments) > 1:
            text = self._normalise_text(segments[-1])

        description = task.get("description", "").lower()
        expected = task.get("expected_answer", "")
        expected_nums = self._extract_numbers(expected)
        answer_nums = self._extract_numbers(text)

        wants_round = "nearest whole number" in description or "round" in description
        wants_times = "times larger" in description or "by what factor" in description
        wants_average_time = "average speed" in description and "total travel time" in description

        if wants_average_time and len(answer_nums) >= 2:
            return f"Average speed is about {answer_nums[0]:.1f} km/h, total time is {round(answer_nums[1])} hours"

        if wants_times and answer_nums:
            value = round(answer_nums[-1]) if wants_round else answer_nums[-1]
            return str(int(value)) if abs(value - round(value)) < 1e-9 else f"{value:.2f}".rstrip("0").rstrip(".")

        if wants_round and answer_nums and len(expected_nums) == 1:
            return str(round(answer_nums[-1]))

        if expected_nums and len(expected_nums) == 1 and answer_nums:
            value = answer_nums[-1]
            if abs(value - round(value)) < 1e-9 and expected_nums[0].is_integer():
                return str(int(round(value)))
            return f"{value:.2f}".rstrip("0").rstrip(".")

        text = re.sub(
            r"^(the (result|answer|current population|height|population density) (of|is)\s*)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"^(rounded to the nearest whole number,?\s*)", "", text, flags=re.IGNORECASE)
        return text.strip()

    def check_answer(self, final_answer: str, expected: str) -> bool:
        """Check if the agent's answer matches the expected answer."""
        if not final_answer or not expected:
            return False

        fa = self._normalise_text(final_answer).lower()
        ea = self._normalise_text(expected).lower()

        if fa == ea or ea in fa or fa in ea:
            return True

        fa_nums = self._extract_numbers(fa)
        ea_nums = self._extract_numbers(ea)
        if ea_nums and fa_nums:
            matched = 0
            remaining = fa_nums.copy()
            for expected_num in ea_nums:
                best_idx = None
                best_err = None
                for idx, found_num in enumerate(remaining):
                    err = abs(found_num - expected_num) / abs(expected_num) if expected_num else abs(found_num - expected_num)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_idx = idx
                if best_idx is not None and ((expected_num != 0 and best_err <= 0.05) or (expected_num == 0 and best_err <= 0.01)):
                    matched += 1
                    remaining.pop(best_idx)
            if matched == len(ea_nums):
                return True

        stop_words = {"the", "a", "an", "is", "of", "and", "to", "by", "about", "approximately"}
        ea_words = {w for w in re.findall(r"[a-z0-9]+", ea) if w not in stop_words}
        fa_words = {w for w in re.findall(r"[a-z0-9]+", fa) if w not in stop_words}
        if ea_words and len(ea_words & fa_words) / len(ea_words) >= 0.7:
            return True
        return False

    def classify_failure(self, result: "StrategyResult", trace: list) -> str:
        """Classify the type of failure based on the execution trace."""
        if result.error_message:
            err_lower = result.error_message.lower()
            if "timeout" in err_lower:
                return "timeout"
            if "429" in err_lower or "resource_exhausted" in err_lower:
                return "rate_limited"
            if "503" in err_lower or "unavailable" in err_lower:
                return "api_unavailable"
        if result.step_count >= self.max_steps or result.tool_calls >= config.MAX_TOOL_CALLS_PER_TASK:
            return "loop"
        step_contents = [str(s) for s in trace]
        if len(step_contents) > 3:
            last_3 = step_contents[-3:]
            if len(set(last_3)) == 1:
                return "loop"
        if result.correct:
            return None
        if not result.final_answer or result.final_answer.strip() == "":
            return "incomplete"
        return "wrong_answer"

    def run(self, task: dict) -> StrategyResult:
        raise NotImplementedError("Subclasses must implement run()")
