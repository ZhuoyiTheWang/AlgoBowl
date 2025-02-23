# Number of test cases to generate
$num_cases = 10
# Fixed seed for the Python generator (seed is not varied here)
$seed = 42

# Function to generate a random floating-point number in a given range
function Get-RandomFloat {
    param (
        [double]$min,
        [double]$max
    )
    return [math]::Round(($min + (Get-Random -Minimum 0.0 -Maximum 1.0) * ($max - $min)), 3)
}

for ($i = 1; $i -le $num_cases; $i++) {
    # Sample R uniformly between 100 and 500
    $R = Get-Random -Minimum 100 -Maximum 501

    # Compute maximum C so that R * C = 100000. Using integer division.
    $C = [math]::Floor(100000 / $R)

    # Sample tree_prob uniformly from [0.1, 0.5]
    $tree_prob = Get-RandomFloat -min 0.1 -max 0.8

    # Sample critical_ratio uniformly from [0.3, 0.7]
    $critical_ratio = Get-RandomFloat -min 0.3 -max 0.7

    # Sample noise uniformly from [0.0, 0.2]
    $noise = Get-RandomFloat -min 0.0 -max 0.2

    # Construct a unique output filename
    $output_file = "input_${R}_${C}_tp${tree_prob}_cr${critical_ratio}_noise${noise}.txt"
    Write-Output "Generating ${output_file} with R=${R}, C=${C} (R*C=$(($R * $C))), tree_prob=${tree_prob}, critical_ratio=${critical_ratio}, noise=${noise}"

    # Run the Python generator (assumed to be generate_hard_input.py in the same directory)
    python generate_hard_input.py $R $C $tree_prob $critical_ratio $noise $seed $output_file
}
