"""
LangChain-compatible tools for the Agentic Planning Study.

These are real tools that interact with actual APIs:
1. calculator — Evaluates math expressions (local, deterministic)
2. web_search — Searches the web via DuckDuckGo (real API)
3. wikipedia_lookup — Fetches summaries from Wikipedia REST API (real API)

All tools use the LangChain @tool decorator for framework compatibility.
"""
import math
import re
import requests
import time
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool 1: Calculator (local — deterministic, no API needed)
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Supports basic arithmetic, powers, sqrt, and common math functions.
    Input should be a math expression string like '2 + 3 * 4' or 'sqrt(144)'.
    """
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow, "log": math.log,
            "log10": math.log10, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
        }
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
        expr = re.sub(r'(\d),(\d)', r'\1\2', expr)
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2: Web Search via DuckDuckGo Instant Answer API
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo for information.
    Input should be a natural language search query like 'population of France 2024'.
    Results may be partial, ambiguous, or unavailable — this is a real API call.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("Abstract"):
            source = data.get("AbstractSource", "Unknown")
            return f"{data['Abstract']} (Source: {source})"

        if data.get("Answer"):
            return f"{data['Answer']}"

        if data.get("RelatedTopics"):
            results = []
            for topic in data["RelatedTopics"][:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(topic["Text"])
            if results:
                return "Related results:\n" + "\n".join(f"- {r}" for r in results)

        if data.get("Infobox") and data["Infobox"].get("content"):
            info_items = []
            for item in data["Infobox"]["content"][:5]:
                if "label" in item and "value" in item:
                    info_items.append(f"{item['label']}: {item['value']}")
            if info_items:
                return "Information found:\n" + "\n".join(info_items)

        return f"No clear results found for: {query}. Try rephrasing your search."

    except requests.exceptions.Timeout:
        return f"Search timed out for query: {query}"
    except requests.exceptions.ConnectionError:
        return f"Connection error — could not reach search API."
    except Exception as e:
        return f"Search error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Wikipedia Lookup via REST API
# ---------------------------------------------------------------------------
@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia and return a summary.
    Input should be a topic name like 'France', 'Speed of light', or 'Alexander Graham Bell'.
    Results may be missing, redirected, or ambiguous — this is a real API call.
    """
    try:
        topic_formatted = topic.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic_formatted}"
        headers = {
            "User-Agent": "AgenticPlanningStudy/1.0 (academic research)",
            "Accept": "application/json",
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 404:
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "opensearch",
                "search": topic,
                "limit": 3,
                "format": "json",
            }
            search_resp = requests.get(search_url, params=search_params,
                                        headers=headers, timeout=10)
            search_data = search_resp.json()
            if len(search_data) > 1 and search_data[1]:
                suggestions = ", ".join(search_data[1][:3])
                return f"No exact article found for '{topic}'. Did you mean: {suggestions}?"
            return f"No Wikipedia article found for: {topic}"

        response.raise_for_status()
        data = response.json()

        title = data.get("title", topic)
        extract = data.get("extract", "")
        description = data.get("description", "")

        if extract:
            if len(extract) > 800:
                extract = extract[:800] + "..."
            result = f"Wikipedia: {title}"
            if description:
                result += f" ({description})"
            result += f"\n{extract}"
            return result

        return f"Wikipedia article '{title}' found but no summary available."

    except requests.exceptions.Timeout:
        return f"Wikipedia lookup timed out for: {topic}"
    except requests.exceptions.ConnectionError:
        return f"Connection error — could not reach Wikipedia."
    except Exception as e:
        return f"Wikipedia lookup error: {str(e)}"


# ---------------------------------------------------------------------------
# Helper: get all tools as a list (LangChain format)
# ---------------------------------------------------------------------------
def get_all_tools() -> list:
    """Return all available tools as a list of LangChain Tool objects."""
    return [calculator, web_search, wikipedia_lookup]
