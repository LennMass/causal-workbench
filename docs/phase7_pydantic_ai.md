# LLM Explanation Agent with Pydantic AI

An agent that interprets causal results in plain language with
structured, validated output.

The LLM doesn't *do* the statistics — DoubleML and/or TabPFN do. The LLM *interprets*
the numbers for a non-technical audience.

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

## Files 
- `src/agents/explainer.py` 
- `src/core/schemas.py` 
- `src/api/main.py` 

# Resource 
[Pydantic AI](https://pydantic.dev/docs/ai/overview/)

