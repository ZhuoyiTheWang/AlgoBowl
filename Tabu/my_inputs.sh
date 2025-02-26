#!/bin/bash

heuristic_folder="../small_outputs"
input_folder="../small_inputs"

for input_file in "$input_folder"/*; do

    filename=$(basename "$input_file")
    heuristic_file="$heuristic_folder/output_$filename"

    if [ -f "$heuristic_file" ]; then
        echo "Running tabu_solver.go with:"
        echo "  -init=$heuristic_file"
        echo "  $input_file"

        go run tabu_solver.go -init="$heuristic_file" "$input_file" >> "../iterationLog.txt"
    else
        echo "Warning: No matching heuristic file found for $input_file"
    fi
done

exit 0
