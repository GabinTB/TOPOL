#!/bin/bash

TIMER_START=$(date +%s)

PROJECT_HOME=$(pwd)
echo "Current project absolute path: $PROJECT_HOME (to be added in the .env file)"

ENV_NAME="topol"
ENV_PATH="$HOME/miniforge3/envs/$ENV_NAME"
PYTHON_VERSION="3.10"

echo "Initializing, updating and starting Conda..."
mamba update conda -y; echo "Conda updated!"
conda init; echo "Conda initialized!"
eval "$(conda shell.bash hook)"; echo "Conda shell initialized!"

echo "Checking if conda is installed and proceeding with installations..."
if [ -d "$ENV_PATH" ]; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping environment creation."

else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    mamba create -n $ENV_NAME python=$PYTHON_VERSION -y; echo "Conda environment '$ENV_NAME' created with Python version $PYTHON_VERSION!"
fi

conda activate $ENV_NAME

if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    echo "!!! Conda environment '$ENV_NAME' is activated, continuing with setup !!!"

    pip3 install --upgrade pip; echo "pip upgraded!"
    pip3 install --no-cache-dir -r .devenv/requirements.txt
    pip3 install -U 'spacy[apple]'
    # pip3 install -U 'spacy[cuda12x]'
    echo "Dependencies installed!"
else
    echo "!!! Conda environment '$ENV_NAME' is not activated, failed to install deps... !!!"
fi

TIME=$(($(date +%s)- $TIMER_START))
HOURS=$((TIME / 3600))
MINUTES=$((TIME % 3600 / 60))
SECONDS=$((TIME % 60))
if [ $HOURS -gt 0 ]; then
    echo "Finished in $HOURS hours, $MINUTES minutes and $SECONDS seconds."
elif [ $MINUTES -gt 0 ]; then
    echo "Finished in $MINUTES minutes and $SECONDS seconds."
else
    echo "Finished in $SECONDS seconds."
fi