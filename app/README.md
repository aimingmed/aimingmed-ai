# How to work with this app repository

Build the images:

```bash
docker compose up --build -d
```

I

# Run the tests for backend:

```bash
docker compose exec backend pipenv run python -m pytest --disable-warnings --cov="."
```

Lint:

```bash
docker compose exec backend pipenv run flake8 .
```

Run Black and isort with check options:

```bash
docker compose exec backend pipenv run black . --check
docker compose exec backend pipenv run isort . --check-only
```

Make code changes with Black and isort:

```bash
docker compose exec backend pipenv run black .
docker compose exec backend pipenv run isort .
```

# Postgres

Want to access the database via psql?

```bash
docker compose exec -it database psql -U postgres
```

Then, you can connect to the database and run SQL queries. For example:

```sql
# \c web_dev
# \dt
```
