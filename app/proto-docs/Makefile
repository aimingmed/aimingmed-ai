lint-check: 
	@poetry run ruff check src/ tests/

lint-fix:
	@poetry run ruff --fix src/ tests/

format-check:
	@poetry run black --check --diff src/ tests/

format-fix:
	@poetry run black src/ tests/

statictypes-check:
	@poetry run pyright src/ tests/

run-tests:
	@poetry run pytest

complete-check: lint-check format-check statictypes-check run-tests
complete-fix: lint-fix format-fix

