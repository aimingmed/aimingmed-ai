# tests

Here, we provide integration tests for (all) components.

These integration tests can be run locally or via docker-compose.

## run using docker-compose

Run all tests using the dummy implementations of backend and jobs:

```bash
docker-compose up -d --build backend frontend
docker-compose up --build tests
```

Stop all containers:

```bash
docker-compose down -v
```

## run locally

You can start the dummy implementations as follows:

```bash
docker-compose up -d --build backend frontend
```

To run the tests locally, you must specify the base urls of backend and jobs api:

```bash
export BACKEND_URL="http://localhost:8004"
export FRONTEND_URL="http://localhost:3000"
```

All values default to `http://localhost:8080`.

### install dependencies

To install the dependencies for the tests, execute the following (in `/app/tests/`):

```bash
pipenv install
```

### run tests

To run the tests locally, execute the following:

```bash
pipenv run pytest tests/integration/
```

To execute tests for jobs / backend only, execute the following:

```bash
pipenv run pytest tests/integration/jobs
pipenv run pytest tests/integration/backend
```

You can also run single groups of tests, e.g.:

```bash
pipenv run pytest tests/integration/backend/tests/test_constraint_types.py -k create
```
