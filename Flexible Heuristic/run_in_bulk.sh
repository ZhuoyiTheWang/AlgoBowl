#!/bin/bash

for i in {994..994}; do
    echo "Running flex_solver.go on input_group$i.txt"
    go run flex_solver.go "../algobowl_inputs/input_group$i.txt" "tree"

    if [ $? -ne 0 ]; then
        echo "Error encountered while processing input_group$i.txt. Exiting."
        exit 1
    fi
done

echo "All tasks completed successfully."
