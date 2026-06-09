"""
Phase 7 — LLM explanation agent with Pydantic AI.

Takes raw CausalResult statistics and generates a structured,
validated interpretation in plain language.

LEARN:
  - Pydantic AI agent definition
  - Structured output: the LLM fills a Pydantic model, not free text
  - System prompts for domain-specific reasoning
  - Tool/function calling: letting the agent decide which estimator to run
  - Combining statistical rigor with accessibility

SETUP:
  Set your API key: export ANTHROPIC_API_KEY="sk-..."
  (or OPENAI_API_KEY if using OpenAI)

TODO (your exercises):
  1. Uncomment and implement the agent below
  2. Add POST /explain to main.py
  3. Try different datasets — does the LLM give sensible interpretations?
  4. Add a "research question" input and let the agent choose the estimator
"""

# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
#
#
# class CausalInterpretation(BaseModel):
#     """Structured LLM output for causal result explanation."""
#
#     summary: str = Field(
#         description="2-3 sentence plain-language summary of the finding"
#     )
#     significance_assessment: str = Field(
#         description="Is this statistically significant? Practically meaningful?"
#     )
#     effect_size_context: str = Field(
#         description="How large is this effect in context?"
#     )
#     threats_to_validity: list[str] = Field(
#         description="Potential issues: unobserved confounders, selection bias, etc."
#     )
#     suggested_robustness_checks: list[str] = Field(
#         description="Next steps: sensitivity analysis, alternative estimators, etc."
#     )
#
#
# SYSTEM_PROMPT = """You are a causal inference expert. Given statistical results
# from a DoubleML estimation, provide a clear, accurate interpretation.
#
# Be precise about what causal claims are and aren't supported.
# Always mention identification assumptions and potential threats.
# Use plain language but don't sacrifice accuracy.
# """
#
#
# explainer_agent = Agent(
#     model="claude-sonnet-4-20250514",
#     system_prompt=SYSTEM_PROMPT,
#     result_type=CausalInterpretation,
# )
#
#
# async def explain_result(
#     estimator: str,
#     learner: str,
#     coefficient: float,
#     std_error: float,
#     ci_lower: float,
#     ci_upper: float,
#     p_value: float,
#     confidence_level: float,
#     treatment_col: str,
#     outcome_col: str,
# ) -> CausalInterpretation:
#     """Run the explanation agent on a causal result."""
#
#     prompt = f"""
#     Interpret these causal inference results:
#
#     Estimator: {estimator}
#     ML Learner: {learner}
#     Treatment variable: {treatment_col}
#     Outcome variable: {outcome_col}
#
#     Estimated ATE: {coefficient:.4f}
#     Standard Error: {std_error:.4f}
#     {confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
#     p-value: {p_value:.6f}
#     """
#
#     result = await explainer_agent.run(prompt)
#     return result.data
