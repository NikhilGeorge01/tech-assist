# FastAPI Introduction

FastAPI is a modern Python web framework used for building APIs quickly and efficiently. It is designed for high performance and supports asynchronous programming.

## Features

- Fast execution speed
- Automatic request validation
- Built-in API documentation
- Async and await support

## Basic Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}
```
