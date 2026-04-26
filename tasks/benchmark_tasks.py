"""
Benchmark tasks for evaluating planning strategies.

Tasks are categorized into three complexity levels:
- Low:    1-2 steps, minimal reasoning, single tool call at most
- Medium: Multi-step reasoning with multiple tool interactions
- High:   Complex decision-making, cross-referencing, iterative reasoning

All tasks require real tool use (web search, Wikipedia, calculator).
Expected answers are approximate — checked with flexible matching.
"""

BENCHMARK_TASKS = [
    # =====================================================================
    # LOW COMPLEXITY — 1-2 steps, direct answers or single tool call
    # =====================================================================
    {
        "id": "low_01",
        "description": "Use Wikipedia to find the capital of Australia.",
        "complexity": "low",
        "domain": "factual_lookup",
        "expected_answer": "Canberra",
        "requires_tools": True,
    },
    {
        "id": "low_02",
        "description": "Use the calculator to compute: 347 * 29 + 156",
        "complexity": "low",
        "domain": "arithmetic",
        "expected_answer": "10219",
        "requires_tools": True,
    },
    {
        "id": "low_03",
        "description": "Search the web to find what element has the chemical symbol 'Au'.",
        "complexity": "low",
        "domain": "factual_lookup",
        "expected_answer": "Gold",
        "requires_tools": True,
    },
    {
        "id": "low_04",
        "description": "Use Wikipedia to find who wrote the novel '1984'.",
        "complexity": "low",
        "domain": "factual_lookup",
        "expected_answer": "George Orwell",
        "requires_tools": True,
    },
    {
        "id": "low_05",
        "description": "Calculate the square root of 2025 using the calculator.",
        "complexity": "low",
        "domain": "arithmetic",
        "expected_answer": "45",
        "requires_tools": True,
    },
    {
        "id": "low_06",
        "description": "Search the web for the boiling point of water in Fahrenheit.",
        "complexity": "low",
        "domain": "factual_lookup",
        "expected_answer": "212",
        "requires_tools": True,
    },

    # =====================================================================
    # MEDIUM COMPLEXITY — Multi-step reasoning, multiple tool interactions
    # =====================================================================
    {
        "id": "med_01",
        "description": (
            "Look up the population of Brazil on Wikipedia, then use the calculator "
            "to find how many times larger Brazil's population is compared to Portugal's "
            "population (approximately 10.3 million). Round to the nearest whole number."
        ),
        "complexity": "medium",
        "domain": "multi_step_qa",
        "expected_answer": "20",
        "requires_tools": True,
    },
    {
        "id": "med_02",
        "description": (
            "Search the web for the height of Mount Everest in meters. "
            "Then use the calculator to convert that height to feet (multiply by 3.281)."
        ),
        "complexity": "medium",
        "domain": "multi_step_qa",
        "expected_answer": "29032",
        "requires_tools": True,
    },
    {
        "id": "med_03",
        "description": (
            "Use Wikipedia to find when the Eiffel Tower was completed. "
            "Then calculate how many years old it is as of 2024."
        ),
        "complexity": "medium",
        "domain": "multi_step_qa",
        "expected_answer": "135",
        "requires_tools": True,
    },
    {
        "id": "med_04",
        "description": (
            "Search for the speed of sound in air (in meters per second). "
            "Then calculate how long it would take (in seconds) for sound to travel "
            "10 kilometers. Round to the nearest whole number."
        ),
        "complexity": "medium",
        "domain": "multi_step_qa",
        "expected_answer": "29",
        "requires_tools": True,
    },
    {
        "id": "med_05",
        "description": (
            "Look up the area of Canada on Wikipedia (in square kilometers). "
            "Then look up the area of the United States. "
            "Which country is larger and by approximately how many million square kilometers?"
        ),
        "complexity": "medium",
        "domain": "multi_step_qa",
        "expected_answer": "Canada is larger by approximately 0.15 million",
        "requires_tools": True,
    },
    {
        "id": "med_06",
        "description": (
            "Search for who invented the World Wide Web. "
            "Then look up that person on Wikipedia to find what year they invented it. "
            "Calculate how many years ago that was from 2024."
        ),
        "complexity": "medium",
        "domain": "multi_step_qa",
        "expected_answer": "Tim Berners-Lee invented it in 1989, which was 35 years ago",
        "requires_tools": True,
    },

    # =====================================================================
    # HIGH COMPLEXITY — Complex reasoning, cross-referencing, iterative steps
    # =====================================================================
    {
        "id": "high_01",
        "description": (
            "Research the following using Wikipedia and web search: "
            "Find the populations of India and China. "
            "Then find the total area (sq km) of each country. "
            "Calculate the population density (people per sq km) for each country "
            "and determine which country is more densely populated and by what factor."
        ),
        "complexity": "high",
        "domain": "multi_step_analysis",
        "expected_answer": "India is more densely populated by a factor of about 3",
        "requires_tools": True,
    },
    {
        "id": "high_02",
        "description": (
            "A train travels from City A to City B at 80 km/h and returns at 60 km/h. "
            "First, search for the formula for average speed of a round trip "
            "(hint: it's the harmonic mean, not the arithmetic mean). "
            "Then use the calculator to compute the average speed for the round trip. "
            "Finally, if the one-way distance is 240 km, calculate the total travel time in hours."
        ),
        "complexity": "high",
        "domain": "math_reasoning",
        "expected_answer": "Average speed is about 68.6 km/h, total time is 7 hours",
        "requires_tools": True,
    },
    {
        "id": "high_03",
        "description": (
            "Look up the GDP (nominal) of Japan, Germany, and the United Kingdom using Wikipedia. "
            "Rank them from highest to lowest. "
            "Calculate what percentage each country's GDP represents of the combined total of all three. "
            "Present the ranking with percentages."
        ),
        "complexity": "high",
        "domain": "multi_step_analysis",
        "expected_answer": "Germany, Japan, UK",
        "requires_tools": True,
    },
    {
        "id": "high_04",
        "description": (
            "Search for the distance from Earth to Mars at its closest approach (in km). "
            "Then search for the speed of the Voyager 1 spacecraft (in km/h). "
            "Calculate how many days it would take Voyager 1 to reach Mars at closest approach. "
            "Then compare: how many days would it take light to reach Mars at closest approach? "
            "(Use speed of light = 300,000 km/s)"
        ),
        "complexity": "high",
        "domain": "math_reasoning",
        "expected_answer": "Voyager 1 would take about 98 days, light takes about 3 minutes",
        "requires_tools": True,
    },
    {
        "id": "high_05",
        "description": (
            "Use Wikipedia to find the year each of these was invented: "
            "the telephone, the light bulb, and the internet. "
            "Calculate the time gap (in years) between each consecutive invention. "
            "Which gap was the longest?"
        ),
        "complexity": "high",
        "domain": "multi_step_analysis",
        "expected_answer": "telephone 1876, light bulb 1879, internet 1983; longest gap is light bulb to internet at about 104 years",
        "requires_tools": True,
    },
    {
        "id": "high_06",
        "description": (
            "Research: What is the tallest building in the world? Find its height in meters. "
            "Then find the tallest building in Europe and its height. "
            "Calculate the difference in height between them. "
            "If you stacked the European building on top of itself, "
            "how many copies would you need to match or exceed the world's tallest?"
        ),
        "complexity": "high",
        "domain": "multi_step_analysis",
        "expected_answer": "Burj Khalifa 828m, Lakhta Center 462m, difference 366m, need 2 copies",
        "requires_tools": True,
    },
]


def get_all_tasks() -> list:
    """Return all benchmark tasks."""
    return BENCHMARK_TASKS


def get_tasks_by_complexity(complexity: str) -> list:
    """Return tasks filtered by complexity level."""
    return [t for t in BENCHMARK_TASKS if t["complexity"] == complexity]
