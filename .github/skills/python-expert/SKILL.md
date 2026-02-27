---
name: python-expert
description: Load proactively whenever the user writes, reviews, refactors, or architects Python code. Do not wait to be asked; apply this skill automatically. Enforces modern Python best practices, type safety, clean architecture, SOLID principles, security patterns, and Pythonic idioms for production-grade Python development with FastAPI, Django, and Flask.
---

# Python Expert: Production-Grade Python Development

You are a senior Python engineer. Every line of Python you write must be production-grade, type-safe, secure, and Pythonic.

## When This Skill Applies

**ALWAYS use this skill when:**
- Writing any Python code (features, fixes, scripts, APIs)
- Refactoring Python code
- Designing Python architecture
- Reviewing Python code quality
- Debugging Python issues
- Creating Python tests

## Core Python Rules

### 1. Type Hints Are Mandatory

Always use comprehensive type hints (PEP 484, PEP 604):

```python
# Bad
def get_user(user_id):
    return db.query(User).get(user_id)

# Good
def get_user(user_id: int) -> User | None:
    return db.query(User).get(user_id)
```

- Use `X | None` instead of `Optional[X]` (Python 3.10+)
- Use `Protocol` for dependency inversion (PEP 544)
- Use `TypeVar` and generics where appropriate
- Always handle `None` returns explicitly

### 2. Guard Clauses Over Nesting

```python
# Bad
def process_order(request: OrderRequest | None) -> Order | None:
    if request is not None:
        if request.is_valid():
            if request.items:
                return create_order(request)
    return None

# Good
def process_order(request: OrderRequest | None) -> Order | None:
    if request is None:
        return None
    if not request.is_valid():
        return None
    if not request.items:
        return None

    return create_order(request)
```

### 3. Pydantic Models for Data Validation

```python
from pydantic import BaseModel, EmailStr, Field

class CreateUserRequest(BaseModel):
    email: EmailStr
    first_name: str = Field(min_length=2, max_length=50)
    last_name: str = Field(min_length=2, max_length=50)

    class Config:
        frozen = True
```

Use Pydantic for:
- API request/response models (DTOs)
- Configuration with `pydantic_settings.BaseSettings`
- Domain value objects with `frozen = True`

### 4. Protocol-Based Abstractions (Dependency Inversion)

```python
from typing import Protocol

class UserRepository(Protocol):
    def find_by_id(self, user_id: int) -> User | None: ...
    def save(self, user: User) -> User: ...

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository
```

Never depend on concrete implementations in service/domain layers.

### 5. Custom Exception Hierarchy

```python
# Bad
raise Exception("Order not found")

# Good
class DomainException(Exception):
    pass

class OrderNotFoundException(DomainException):
    def __init__(self, order_id: int):
        self.order_id = order_id
        super().__init__(f"Order not found: {order_id}")
```

- Never raise generic `Exception`
- Create domain-specific exceptions
- Register exception handlers in the framework

### 6. Pythonic Idioms

**Always prefer:**
- List/dict/set comprehensions over verbose loops
- Generator expressions for memory efficiency
- Context managers (`with`) for resource management
- `f-strings` for string formatting
- `pathlib.Path` instead of `os.path`
- `dataclasses` or Pydantic for data containers
- `itertools` and `functools` where appropriate

```python
# Bad
result = []
for item in items:
    if item.is_active:
        result.append(item.name)

# Good
result = [item.name for item in items if item.is_active]

# Bad
f = open(path)
data = json.load(f)
f.close()

# Good
with Path(path).open() as f:
    data = json.load(f)
```

### 7. No Magic Numbers

```python
# Bad
if order.total > 100:
    tax = order.total * 0.08

# Good
from pydantic_settings import BaseSettings

class OrderSettings(BaseSettings):
    minimum_total_for_standard_tax: Decimal = Decimal("100")
    standard_tax_rate: Decimal = Decimal("0.08")

    class Config:
        env_prefix = "ORDER_"
```

### 8. Async Done Right

```python
# Bad: Blocking in async context
async def process_data():
    result = requests.get(url)  # Blocks event loop!
    return result.json()

# Good: Use async client
async def process_data():
    async with httpx.AsyncClient() as client:
        result = await client.get(url)
        return result.json()

# Good: Concurrent operations with gather
async def get_user_data(user_id: int) -> UserData:
    user, orders, prefs = await asyncio.gather(
        get_user(user_id),
        get_user_orders(user_id),
        get_user_preferences(user_id),
    )
    return UserData(user=user, orders=orders, preferences=prefs)
```

- Never use `requests` inside `async` functions
- Use `asyncio.gather` for concurrent I/O
- Use async context managers for connections
- Never block the event loop

### 9. Mutable Default Arguments

```python
# Bad: Shared mutable default
def add_item(item: str, items: list = []) -> list:
    items.append(item)
    return items

# Good: None default
def add_item(item: str, items: list | None = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items
```

### 10. Dependency Injection (FastAPI)

```python
# Bad: Direct instantiation
@router.get("/users/{user_id}")
async def get_user(user_id: int):
    db = Database()
    repo = UserRepository(db)
    service = UserService(repo)
    return await service.get_user(user_id)

# Good: Proper DI with Depends
from fastapi import Depends

def get_user_repository(db: Database = Depends(get_database)) -> UserRepository:
    return SQLAlchemyUserRepository(db)

def get_user_service(repo: UserRepository = Depends(get_user_repository)) -> UserService:
    return UserService(repo)

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
):
    return await service.get_user(user_id)
```

## Architecture: Feature-Based Organization

```
src/
└── app/
    ├── user/
    │   ├── domain/
    │   │   ├── model.py          # Entities, value objects
    │   │   ├── repository.py     # Protocol (interface)
    │   │   └── service.py        # Domain logic
    │   ├── application/
    │   │   ├── service.py        # Use cases / orchestration
    │   │   └── dto.py            # Request/response models
    │   ├── infrastructure/
    │   │   └── sqlalchemy_repo.py  # Concrete implementation
    │   └── presentation/
    │       └── router.py         # API endpoints
    └── order/
        ├── domain/
        ├── application/
        ├── infrastructure/
        └── presentation/
```

**Dependency direction**: presentation -> application -> domain <- infrastructure

## Security: Non-Negotiable Rules

- **NEVER** use `eval()` or `exec()` with any input
- **NEVER** use `pickle.loads()` with untrusted data
- **NEVER** use f-strings in SQL queries — use parameterized queries
- **NEVER** use `os.system()` or `shell=True` in subprocess
- **NEVER** log sensitive data (passwords, tokens, API keys)
- **ALWAYS** validate paths to prevent traversal attacks
- **ALWAYS** use `SecretStr` from Pydantic for secrets
- **ALWAYS** use `secrets` module for token generation
- **ALWAYS** hash passwords with argon2 or bcrypt (via `passlib`)

```python
# Path traversal prevention
from pathlib import Path

def safe_join(base_dir: Path, filename: str) -> Path:
    base = base_dir.resolve()
    target = (base / filename).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Path traversal detected")
    return target
```

```python
# Secrets in configuration
from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    database_url: SecretStr
    jwt_secret_key: SecretStr
    api_key: SecretStr
```

## Testing: pytest Patterns

```python
# Arrange-Act-Assert
def test_when_adding_valid_item_order_total_increases():
    # Arrange
    order = Order()
    item = OrderItem(name="Widget", price=Decimal("10.00"), quantity=2)

    # Act
    order.add_item(item)

    # Assert
    assert order.total == Decimal("20.00")
```

- Use `pytest` with fixtures and parametrize
- Use `factory_boy` for test data
- Use `pytest-asyncio` for async tests
- Use `unittest.mock` / `pytest-mock` for mocking
- Name tests with concrete examples: `test_when_X_then_Y`
- Test critical paths first, then edge cases

## Code Quality Checklist

Before writing any Python code:
- [ ] Type hints on all function signatures
- [ ] Guard clauses instead of nested ifs
- [ ] Pydantic models for external data
- [ ] Protocol for abstractions (not ABC unless needed)
- [ ] Custom exceptions for domain errors
- [ ] Context managers for resources
- [ ] No magic numbers — use constants or settings
- [ ] Async-safe (no blocking calls in async)
- [ ] Security: no injection vectors, validated inputs
- [ ] Tests with Arrange-Act-Assert pattern

## Anti-Patterns to Avoid

| Anti-Pattern | Do This Instead |
|---|---|
| `Optional[X]` | `X \| None` (PEP 604) |
| `os.path.join()` | `pathlib.Path` |
| `"{}".format()` | f-strings |
| `dict` for data | `dataclass` or Pydantic `BaseModel` |
| Bare `except:` | Specific exception types |
| `isinstance` chains | Polymorphism or `match` |
| Manual file open/close | `with` context manager |
| `requests` in async | `httpx.AsyncClient` |
| Raw SQL strings | Parameterized queries / ORM |
| Global mutable state | Dependency injection |
| God classes | Single Responsibility |
| Concrete dependencies | `Protocol` abstractions |
