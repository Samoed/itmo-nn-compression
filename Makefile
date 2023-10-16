# https://github.com/samuelcolvin/pydantic/blob/master/Makefile
.DEFAULT_GOAL := all
folders = baseline/ prune/ quantization/ distil/ different_frameworks/ cluster/
files = update_table.py utils.py
poetry = poetry run
isort = isort
black = black
mypy = mypy
flake8  = flake8
pyupgrade = pyupgrade --py310-plus

.PHONY: install-linting
install-linting:
	poetry add flake8 black isort mypy pyupgrade -G dev

.PHONY: format
format:
	$(poetry) $(pyupgrade)
	$(poetry) $(isort) $(folders) $(files)
	$(poetry) $(black) $(folders) $(files)
	# $(poetry) $(mypy) $(folders)
	$(poetry) $(flake8) $(folders)

.PHONY: mlflow
mlflow:
	cd mlflow && docker compose up mlflow -d

.PHONY: all
all: format
