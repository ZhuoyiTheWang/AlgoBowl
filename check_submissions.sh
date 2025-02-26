#!/bin/bash

submissions_folder="./submitted_outputs"
input_folder="./algobowl_inputs"
combined_results="./submission_verification_results.txt"

if [ -f "$combined_results" ]; then
    rm "$combined_results"
fi

touch "$combined_results"

for submission_file in "$submissions_folder"/*; do
    filename=$(basename "$submission_file")
    echo "Checking ${submission_file}"
    input_number="${filename:19}"
    input_file="${input_folder}/input_group${input_number}"
    python ./Python_Verification.py "$input_file" "$submission_file" >> "$combined_results" 2>&1
    
done

exit 0
