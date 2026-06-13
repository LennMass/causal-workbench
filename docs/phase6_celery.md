# Async Jobs with Celery

Move heavy computation to background workers so the API responds instantly.

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

## Files
- `src/api/tasks.py` 
- `src/api/main.py` 


## Resources
- [Redis](https://github.com/redis/redis)
- Install options
    - **Mac:** `brew install redis`
    - **Linux:** `apt install redis-server`
    - **Docker:** `docker run -d -p 6379:6379 redis`
