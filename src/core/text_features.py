"""
Phase 5 — Text confounder embeddings with Transformers.

Use sentence-transformers to embed free-text columns into numeric vectors,
then concatenate with structured features for DoubleML.

LEARN:
  - sentence-transformers SentenceTransformer model loading
  - Encoding text to fixed-size vectors
  - Integrating embeddings into the Polars pipeline
  - Why text confounders matter for causal identification

TODO (your exercises):
  1. Implement embed_text_column below
  2. Add it to pipeline.py as an optional step
  3. Create a sample dataset with a text column (e.g. product descriptions)
  4. Run estimation with and without text features — does the ATE change?

EXAMPLE DATASET IDEA:
  Estimate effect of a discount (treatment) on purchase amount (outcome),
  with product_description as a text confounder. Without encoding the text,
  you have omitted variable bias if description predicts both treatment
  assignment and outcome.
"""

# from sentence_transformers import SentenceTransformer
# import polars as pl
# import numpy as np


# # Use a small, fast model — runs on CPU in seconds
# DEFAULT_MODEL = "all-MiniLM-L6-v2"


# def embed_text_column(
#     df: pl.DataFrame,
#     text_col: str,
#     model_name: str = DEFAULT_MODEL,
#     prefix: str = "emb_",
# ) -> pl.DataFrame:
#     """
#     Replace a text column with its sentence embedding dimensions.
#
#     Args:
#         df: Polars DataFrame with a text column
#         text_col: name of the column containing text
#         model_name: sentence-transformer model to use
#         prefix: prefix for the new embedding columns
#
#     Returns:
#         DataFrame with text_col removed, embedding columns added
#     """
#     model = SentenceTransformer(model_name)
#     texts = df[text_col].to_list()
#     embeddings = model.encode(texts, show_progress_bar=False)
#
#     # Create new columns: emb_0, emb_1, ..., emb_383
#     emb_df = pl.DataFrame({
#         f"{prefix}{i}": embeddings[:, i]
#         for i in range(embeddings.shape[1])
#     })
#
#     return pl.concat([df.drop(text_col), emb_df], how="horizontal")
