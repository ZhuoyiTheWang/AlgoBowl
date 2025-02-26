#!/bin/bash

# Define the array of input numbers
inputs=(964)  # Add more numbers as needed

for i in "${inputs[@]}"; do
    if [ "$i" -eq 964 ]; then
        init_file="outputs/output_tabu_input_group$i.txt"
    elif [ "$i" -eq 994 ]; then
        init_file="outputs/output_tabu_input_172_581_tp0.126_cr0.586_noise0.155.txt"
    else
        init_file="../Heuristic/outputs/output_input_group$i.txt"
    fi

    echo "Running tabu_solver.go on input_group$i.txt with init $init_file"

    go run tabu_solver.go -init="$init_file" "../algobowl_inputs/input_group$i.txt"

    if [ $? -ne 0 ]; then
        echo "Error encountered while processing input_group$i.txt. Exiting."
        exit 1
    fi
done
