#!/bin/bash

# Loop through all files in the folder that match the pattern "output_from_994_to_****.txt"
for file in output_from_994_to_*.txt; do
  # Extract the number part (****) from the original filename
  number=$(echo "$file" | sed -E 's/.*output_from_994_to_([0-9]{3,4})\.txt/\1/')
  
  # Create the new filename with the format "output_input_group****.txt"
  new_filename="output_input_group${number}.txt"
  
  # Rename the file
  mv "$file" "$new_filename"
  
  echo "Renamed $file to $new_filename"
done
