#!/bin/bash

# Validate input argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 [telco|swat]"
    exit 1
fi

DATASET=$1

# Validate dataset choice
if [[ "$DATASET" != "telco" && "$DATASET" != "swat" ]]; then
    echo "Invalid dataset. Choose 'telco' or 'swat'"
    exit 1
fi

# Create logs directory if not exists
mkdir -p logs_predict

# Run predictions in parallel with logging and track PIDs
declare -A pids
models=("gru" "gcn" "gdn" "mtad_gat")
for model in "${models[@]}"; do
    python models/predict.py \
        --dataset "$DATASET" \
        --model "$model" \
        --params_file "models/${model}/params_${DATASET}.yaml" \
        $([ "$DATASET" = "telco" ] && echo "-iw 205" || echo "-iw 256") > "logs_predict/${model}_${DATASET}.log" 2>&1 &
    pids[$model]=$!
done

# Wait for all background jobs and check their exit status
failed_models=()
for model in "${!pids[@]}"; do
    pid=${pids[$model]}
    wait "$pid" || failed_models+=("$model")
done

# Report any failures
if [ ${#failed_models[@]} -ne 0 ]; then
    echo "The following models failed: ${failed_models[*]}"
    exit 1
else
    echo "All models completed successfully."
fi
