#!/bin/bash

DATASET="FREIHAND-VAL"
LOG_FILE="throughput.txt"
NUM_RUNS=10
R=1

# Clear old log file
> $LOG_FILE

echo "\nRunning eval.py $NUM_RUNS times..."

for i in $(seq 1 $NUM_RUNS)
do
    echo "Run $i/$NUM_RUNS..."
    # Run eval.py and capture its output
    OUTPUT=$(python eval.py --dataset $DATASET --r $R)

    # Extract the number after 'Average throughput:'
    THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Image per second: \s*\K[0-9.]+')

    echo "$THROUGHPUT" >> $LOG_FILE
done

echo "Done. Results saved to $LOG_FILE"

