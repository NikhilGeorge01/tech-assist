# FastAPI Routing

Routing in FastAPI is handled using decorators such as `@app.get()` and `@app.post()`.

## Path Parameters

Path parameters allow values to be passed through the URL.

Example:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```
