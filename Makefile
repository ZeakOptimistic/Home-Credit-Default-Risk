.PHONY: format lint clean run

# End-to-end training and submission execution
run:
	python main.py

# Code formatting and linting
format:
	black src notebooks
	isort src notebooks

lint:
	ruff check src notebooks

# Clean caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
