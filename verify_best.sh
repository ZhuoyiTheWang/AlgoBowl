#!/bin/bash

best_folder="./all_outputs/best_performance"
input_folder="./algobowl_inputs"
combined_results="./submission_verification_results.txt"

if [ -f "$combined_results" ]; then
    rm "$combined_results"
fi

touch "$combined_results"

for input_file in "$input_folder"/*; do
    filename=$(basename "$input_file")
    echo "Checking ${input_file}"
    
    best_file="${best_folder}/output_${filename}"
    python ./Python_Verification.py "$input_file" "$best_file" >> "$combined_results" 2>&1
    
done

exit 0
