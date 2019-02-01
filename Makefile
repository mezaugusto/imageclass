help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean all the cache and log files
	find . -type d -name '__pycache__' -delete
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.log' -delete

install: 
	pip install -r requirements.txt ## Install dependencies from requirements.txt

run: ## Run project
	gunicorn server:app