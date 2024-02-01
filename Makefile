source = canonical_sets

.PHONY: clean format check test install docs

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage

format: clean
	poetry run python -m isort .
	poetry run python -m black .
	make clean

check: clean
	poetry run python -m isort . -c
	poetry run python -m black . --check
	poetry run python -m flake8 .
	poetry run python -m bandit .
	poetry run python -m mypy $(source)
	make clean

test: clean
	poetry run python -m pytest --cov-report term-missing --cov=$(source)
	make clean

install:
	poetry install

docs: 
	cd docs; poetry run make clean html; cd ..; make clean