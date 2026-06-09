# FastAPI API Layer

Serve causal pipeline as a REST API with auto-generated docs.

- `FastAPI()` app, `@app.get()` / `@app.post()` decorators
- `UploadFile` for file uploads
- Pydantic models as `response_model` — auto-validation and docs
- Swagger UI at `/docs`, ReDoc at `/redoc`
- Error handling with `HTTPException`

## How to run
```bash
uvicorn src.api.main:app --reload
```
Then open http://localhost:8000/docs

## Files 
- `src/api/main.py` 


## Resources
- [FastAPI](https://fastapi.tiangolo.com/)

