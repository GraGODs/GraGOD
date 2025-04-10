#!/bin/bash

# Validate input argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 [telco|swat]"
    exit 1
fi

DATASET=$1

# Validate dataset choice
if [[ "$DATASET" != "telco" && "$DATASET" != "swat" && "$DATASET" != "ute" ]]; then
    echo "Invalid dataset. Choose 'telco', 'swat' or 'ute'"
    exit 1
fi

# Create logs directory if not exists
mkdir -p logs

# Run predictions sequentially with logging
models=("gru" "gcn" "gdn" "mtad_gat")
for model in "${models[@]}"; do
    python models/train.py \
        --dataset "$DATASET" \
        --model "$model" \
        --params_file "models/${model}/params_${DATASET}.yaml" > "logs_train/${model}_${DATASET}.log" 2>&1
done

# Wait for all background jobs to complete
wait

