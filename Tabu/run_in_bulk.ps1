# Define the array of input numbers
$inputs = @(977, 995, 1019, 981, 979, 983, 991, 969, 1012, 967, 980, 965, 1021, 971)  # Add more numbers as needed

foreach ($i in $inputs) {
 
    $initFile = "../Heuristic/outputs/output_input_group$i.txt"
    

    Write-Host "Running tabu_solver.go on input_group$i.txt with init $initFile"

    $process = Start-Process -NoNewWindow -Wait -PassThru -FilePath "go" -ArgumentList "run tabu_solver.go -init=$initFile ../algobowl_inputs/input_group$i.txt"

    if ($process.ExitCode -ne 0) {
        Write-Host "Error encountered while processing input_group$i.txt. Exiting."
        exit 1
    }
}

Write-Host "All tasks completed successfully."
