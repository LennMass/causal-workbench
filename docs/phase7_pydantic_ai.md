# Phase 7 — LLM Explanation Agent with Pydantic AI

## Goal
Build an agent that interprets causal results in plain language with
structured, validated output.

## What to learn
- Pydantic AI `Agent` definition with `result_type`
- System prompts for domain expertise
- Structured output: LLM fills a Pydantic model, not free text
- Chaining: statistical results → LLM interpretation → validated response

## Setup
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Files to work on
- `src/agents/explainer.py` — uncomment and implement
- `src/core/schemas.py` — uncomment `CausalInterpretation`
- `src/api/main.py` — add `POST /explain` endpoint

## Exercises
1. Uncomment the agent in `explainer.py`
2. Test it standalone: pass in a `CausalResult` and print the interpretation
3. Add `POST /explain` to the API: runs analysis + explanation in one call
4. Try the "estimator chooser": describe a research question in natural language,
   let the agent decide PLR vs IRM based on the description
5. Add a `threats_to_validity` field and see if the LLM catches real issues

## Key insight
The LLM doesn't *do* the statistics — DoubleML does. The LLM *interprets*
the numbers for a non-technical audience. This separation is important:
never let the LLM make up statistics.
