import os


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


EXECUTION_MODE = os.getenv("EXECUTION_MODE", "async")   # "async" | "sync"
MLFLOW_ENABLED = _flag("MLFLOW_ENABLED", "true")
ENABLE_TABPFN = _flag("ENABLE_TABPFN", "true")
MAX_SYNC_ROWS = int(os.getenv("MAX_SYNC_ROWS", "20000"))

AVAILABLE_LEARNERS = ["sklearn", "xgboost"]
if ENABLE_TABPFN:
    AVAILABLE_LEARNERS.append("tabpfn")