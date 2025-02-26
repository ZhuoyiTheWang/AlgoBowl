import os
import re
import shutil

def main():
    # Path to your main directory that contains the subdirectories (e.g. adjacency, future, etc.)
    main_dir = "all_outputs/"
    
    # Directory to store the best performing files
    best_dir = os.path.join(main_dir, "best_performance")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    
    # Dictionary to hold performances:
    # {
    #   group_number: {
    #       method_name: {
    #           "performance": float,
    #           "filepath": str
    #       }
    #   }
    # }
    performances = {}
    
    # Go through each subdirectory in main_dir
    for method_name in os.listdir(main_dir):
        sub_path = os.path.join(main_dir, method_name)
        
        # Skip anything that isn't a directory or is our 'best_performance' directory
        if not os.path.isdir(sub_path) or method_name == "best_performance":
            continue
        
        # Iterate over files in this subdirectory
        for filename in os.listdir(sub_path):
            # We are looking for files that start with "output" and end with something like "groupXYZ"
            # Example patterns: output_input_group963, output_tabu_input_group963, etc.
            # Use a regex to find the group number
            # e.g., if filename = "output_tabu_input_group963"
            # we'll look for the pattern "group(\d+)"
            match = re.search(r"group(\d+)", filename)
            if match:
                group_number = match.group(1)
                
                file_path = os.path.join(sub_path, filename)
                
                # Read the first line to get performance
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    try:
                        performance_value = float(lines[0].strip())
                    except ValueError:
                        # If the first line isn't a valid number, skip
                        continue
                
                # Store the performance in our dictionary
                if group_number not in performances:
                    performances[group_number] = {}
                
                performances[group_number][method_name] = {
                    "performance": performance_value,
                    "filepath": file_path
                }
    
    # Now we have a dictionary of all performances by group and method
    # We can write them to a summary text file and pick the best
    summary_file = os.path.join(main_dir, "performance_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as summary:
        summary.write("Performance Summary\n")
        summary.write("===================\n\n")
        
        # Sort group numbers so the summary is in ascending group order (optional)
        for group_number in sorted(performances.keys(), key=lambda x: int(x)):
            summary.write(f"Group {group_number}:\n")
            
            method_info = performances[group_number]
            
            # Determine the best performance (lowest) for this group
            best_method = None
            best_value = None
            
            # Write all method performances
            for method_name, perf_data in method_info.items():
                val = perf_data["performance"]
                summary.write(f"  Method: {method_name}, Performance: {val}\n")
                
                if best_value is None or val < best_value:
                    best_value = val
                    best_method = method_name
            
            summary.write(f"--> Best Method: {best_method} (Performance: {best_value})\n\n")
            
            # Copy the best file to best_performance directory
            best_file_path = method_info[best_method]["filepath"]
            
            # Construct a name for the copied file, or just reuse the same filename
            original_filename = os.path.basename(best_file_path)
            destination_path = os.path.join(best_dir, original_filename)
            
            shutil.copy2(best_file_path, destination_path)
    
    print("Summary written to:", summary_file)
    print("Best files copied to:", best_dir)

if __name__ == "__main__":
    main()
