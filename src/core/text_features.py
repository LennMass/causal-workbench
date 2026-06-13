"""
Text confounder embeddings with Transformers.

Use sentence-transformers to embed free-text columns into numeric vectors,
then concatenate with structured features for DoubleML.

- sentence-transformers SentenceTransformer model loading
- Encoding text to fixed-size vectors
- Integrating embeddings into the Polars pipeline


EXAMPLE DATASET IDEA:
  Estimate effect of a discount (treatment) on purchase amount (outcome),
  with product_description as a text confounder.
"""

from sentence_transformers import SentenceTransformer
import polars as pl
import numpy as np


# Use a small, fast model — runs on CPU in seconds
DEFAULT_MODEL = "all-MiniLM-L6-v2"


from sklearn.decomposition import PCA

def embed_text_column(
    df: pl.DataFrame,
    text_col: str,
    model_name: str = DEFAULT_MODEL,
    prefix: str = "emb_",
    n_components: int | None = 10,
) -> pl.DataFrame:
    model = SentenceTransformer(model_name)
    texts = df[text_col].to_list()
    embeddings = model.encode(texts, show_progress_bar=False)

    # PCA for component reduction
    if n_components and n_components < embeddings.shape[1]:
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)

    emb_df = pl.DataFrame({
        f"{prefix}{i}": embeddings[:, i]
        for i in range(embeddings.shape[1])
    })

    return pl.concat([df.drop(text_col), emb_df], how="horizontal")

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  """Compare ATE with and without text confounders."""

  import polars as pl
  from src.core.pipeline import clean_data, prepare_for_doubleml
  from src.core.text_features import embed_text_column
  from src.core.estimators import run_estimation

  DATA_PATH = "data/sample_data_text.csv"

  # --- WITHOUT text features ---
  df_no_text = pl.read_csv(DATA_PATH).drop("product_description")
  df_no_text = clean_data(df_no_text)
  data_no_text = prepare_for_doubleml(df_no_text, "treatment", "outcome")

  result_no_text = run_estimation(data_no_text, estimator="plr", learner="sklearn")


  # --- WITH text features ---
  df_with_text = pl.read_csv(DATA_PATH)
  df_with_text = embed_text_column(df_with_text, "product_description")
  df_with_text = clean_data(df_with_text)
  data_with_text = prepare_for_doubleml(df_with_text, "treatment", "outcome")

  result_with_text = run_estimation(data_with_text, estimator="plr", learner="sklearn")


  print("=== WITHOUT text features ===")
  print(f"ATE: {result_no_text.coefficient:.4f}")
  print(f"95% CI: [{result_no_text.ci_lower:.4f}, {result_no_text.ci_upper:.4f}]")
  print(f"p-value: {result_no_text.p_value:.6f}")

  print("\n=== WITH text features ===")
  print(f"ATE: {result_with_text.coefficient:.4f}")
  print(f"95% CI: [{result_with_text.ci_lower:.4f}, {result_with_text.ci_upper:.4f}]")
  print(f"p-value: {result_with_text.p_value:.6f}")

  print(f"\nDifference in ATE: {abs(result_with_text.coefficient - result_no_text.coefficient):.4f}")
  print(f"Features without text: {data_no_text.n_features}")
  print(f"Features with text: {data_with_text.n_features}")

