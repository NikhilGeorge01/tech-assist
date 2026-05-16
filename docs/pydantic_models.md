# Pydantic Models

Pydantic is a Python library used for data validation and settings management using Python type hints.

## Example Model

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
```
