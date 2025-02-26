#!/bin/bash

for i in {994..994}; do
    echo "Running heuristic_solver.go on input_group$i.txt"
    go run constraint_solver.go "../algobowl_inputs/input_group$i.txt"

    if [ $? -ne 0 ]; then
        echo "Error encountered while processing input_group$i.txt. Exiting."
        exit 1
    fi
done

echo "All tasks completed successfully."
