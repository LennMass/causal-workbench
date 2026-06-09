# Pydantic Schemas


Define strict typed data contracts for every piece of data flowing through the system.

Pydantic solves the problem that Python is dynamically typed. If one passes a string where a number is expected and Python wouldn't complain until something breaks deep in the code. Pydantic catches that at the boundary the moment data enters the system.

FastAPI will use models as request/response types.
We write the schema once, and get validation + docs + serialization for free.

- `BaseModel` with type annotations
- `Field()` for defaults, constraints (`ge`, `le`), and descriptions
- `str` Enums for controlled vocabularies
- `.model_dump()` → dict, `.model_validate()` → model from dict
- Automatic validation: wrong types raise `ValidationError`

## Files 
- `src/core/schemas.py` 

## Resources
- [Pydantic Docs](https://pydantic.dev/docs/validation/latest/get-started/)


