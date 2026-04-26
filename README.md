# Agentic Planning Study

This repository contains the code for my dissertation experiments comparing different agentic planning strategies (ReAct, Plan-and-Solve, and Tree-of-Thoughts).

## Setup

1. Create a virtual environment:
   `python -m venv venv`
2. Activate your environment:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
3. Install requirements:
   `pip install -r requirements.txt`
4. Set up environment variables:
   Copy `.env.example` to `.env` and add your API keys.

## Running the Code

To run the main experiment:
`python run_experiment.py`

If you want to do a quick test run to make sure everything works:
`python run_experiment.py --quick`

You can also run a specific strategy:
`python run_experiment.py --strategy ReAct`

## Results
The results and graphs are automatically saved in the `results/` folder after the experiment finishes running.
# implementation-with-external-api-s
