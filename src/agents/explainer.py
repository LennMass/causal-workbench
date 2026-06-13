"""
LLM explanation agent with Pydantic AI.

The LLM interprets statistical results — it never computes them.
DoubleML or TabPFN does the math, the agent explains it.

- Pydantic AI agent definition
- Structured output: the LLM fills a Pydantic model, not free text
- System prompts for domain-specific reasoning
- Tool/function calling: letting the agent decide which estimator to run
- Combining statistical rigor with accessibility

SETUP:
  Set your API key: export ANTHROPIC_API_KEY="sk-..."
  (or OPENAI_API_KEY if using OpenAI)
"""



from pydantic_ai import Agent
from src.core.schemas import CausalInterpretation


SYSTEM_PROMPT = """You are a causal inference expert who explains statistical 
results to non-technical stakeholders.

RULES:
- Never invent or modify numbers. Only interpret what you are given.
- Be precise about what causal claims ARE and ARE NOT supported.
- Always mention the key identification assumption: no unobserved confounders.
- Distinguish statistical significance from practical significance.
- Use plain language but don't sacrifice accuracy.
- When discussing threats to validity, be specific to the analysis described.
- For robustness checks, suggest concrete, actionable steps.
"""


explainer_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    system_prompt=SYSTEM_PROMPT,
    output_type=CausalInterpretation,
)


async def explain_result(
    estimator: str,
    learner: str,
    coefficient: float,
    std_error: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    confidence_level: float,
    treatment_col: str,
    outcome_col: str,
    n_obs: int,
    n_features: int,
) -> CausalInterpretation:
    """Run the explanation agent on a causal result."""

    prompt = f"""Interpret these causal inference results:

Estimator: {estimator}
ML Learner for nuisance parameters: {learner}
Number of observations: {n_obs}
Number of features/confounders: {n_features}

Treatment variable: {treatment_col}
Outcome variable: {outcome_col}

Estimated Average Treatment Effect (ATE): {coefficient:.4f}
Standard Error: {std_error:.4f}
{confidence_level*100:.0f}% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
p-value: {p_value:.6f}
Statistically significant at {(1-confidence_level)*100:.0f}% level: {"Yes" if p_value < (1 - confidence_level) else "No"}
"""

    result = await explainer_agent.run(prompt)
    return result.output


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def test():
        interpretation = await explain_result(
            estimator="PLR",
            learner="sklearn",
            coefficient=350.42,
            std_error=45.23,
            ci_lower=261.77,
            ci_upper=439.07,
            p_value=0.000001,
            confidence_level=0.95,
            treatment_col="job_training",
            outcome_col="monthly_income",
            n_obs=5000,
            n_features=15,
        )

        print("=== SUMMARY ===")
        print(interpretation.summary)
        print("\n=== SIGNIFICANCE ===")
        print(interpretation.significance_assessment)
        print("\n=== EFFECT SIZE ===")
        print(interpretation.effect_size_context)
        print("\n=== THREATS TO VALIDITY ===")
        for t in interpretation.threats_to_validity:
            print(f"  - {t}")
        print("\n=== ROBUSTNESS CHECKS ===")
        for r in interpretation.suggested_robustness_checks:
            print(f"  - {r}")

    asyncio.run(test())