VENV = .venv
PYTHON = /opt/homebrew/bin/python3.13
PY = $(VENV)/bin/python # Using this as the virtual environment cannot be activated in from the makefile
PIP = $(VENV)/bin/pip # Same as above

MAIN = main.py
SRC = src
DATA_DIR = assets
DATA_ZIP = asl-dataset.zip
DATA = asl-dataset
MODEL_DIR = models

file ?= $(SRC)/$(MAIN)

setup: dirs $(VENV) # data ## Create the directories, a virtual environment

dirs: ## Create the directories
	@mkdir -p $(SRC) $(DATA_DIR) $(MODEL_DIR)

run: | $(SRC) ## Run the application
	@$(PY) $(file)

install: requirements.txt ## Install dependencies
	@$(PIP) install -r $<

uninstall: ## Uninstall dependencies
	@$(PIP) freeze | xargs pip uninstall -y

freeze: ## Freeze the dependencies
	@$(PIP) freeze > requirements.txt

venv: ## Create if it doesn't exist and activate a virtual environment. Use source $(VENV)/bin/activate to activate it.
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created."

# curl -L -o ~/Downloads/asl-dataset.zip\
#   https://www.kaggle.com/api/v1/datasets/download/ayuraj/asl-dataset
data: | $(DATA_DIR)  ## Download the data
	@if [ ! -d "$(DATA_DIR)/$(DATA)" ]; then \
		echo "Downloading the data..."; \
		curl -L -o $(DATA_DIR)/$(DATA_ZIP) 'https://www.kaggle.com/api/v1/datasets/download/ayuraj/asl-dataset'; \
		echo "Unzipping the data..."; \
		cd $(DATA_DIR) && @unzip $(DATA_ZIP); \
		echo "Data downloaded and unzipped at '$(DATA_DIR)/$(DATA)'."; \
	else \
		echo "Data already exists at '$(DATA_DIR)/$(DATA)'."; \
	fi

# pythonpath: ## Set the PYTHONPATH
# 	@export PYTHONPATH="$(shell pwd):$(PYTHONPATH)"
	
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make [target]\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

clean:
	@rm -rf $(VENV)

clear-cache: ## Clear the pip cache
	@pip cache purge