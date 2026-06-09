# Phase 6 — Async Jobs with Celery

## Goal
Move heavy computation to background workers so the API responds instantly.

## What to learn
- Celery app config with Redis broker
- `@app.task` decorator
- Task states: PENDING → STARTED → SUCCESS / FAILURE
- `AsyncResult` for polling
- Running workers alongside the API

## Setup
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Celery worker
celery -A src.api.tasks worker --loglevel=info

# Terminal 3: FastAPI
uvicorn src.api.main:app --reload
```

## Files to work on
- `src/api/tasks.py` — uncomment and implement
- `src/api/main.py` — add async endpoints

## Exercises
1. Uncomment the Celery task in `tasks.py`
2. Add `POST /analyze/async` → save file, dispatch task, return `{job_id}`
3. Add `GET /results/{job_id}` → check task state, return result if done
4. Submit 3 analyses at once and watch the Celery worker process them
5. Add error handling: what happens if the task fails?

## Redis install options
- **Mac:** `brew install redis`
- **Linux:** `apt install redis-server`
- **Docker:** `docker run -d -p 6379:6379 redis`
