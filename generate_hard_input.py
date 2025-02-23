import random
import sys
import numpy as np

def generate_input(R, C, tree_prob, critical_ratio=0.5, noise=0.1, seed=None):
    if seed is not None:
        random.seed(seed)

    grid = []
    row_counts = []
    col_counts = []

    for i in range(R):
        row = []
        for j in range(C):
            cell = 'T' if random.random() < np.random.normal(tree_prob, (1-tree_prob) / 4) else '.'
            row.append(cell)
        grid.append(row)
        
        blank_count = row.count('.')
        base = blank_count * critical_ratio
        offset = int(random.uniform(-noise, noise) * blank_count)
        target = int(round(base)) + offset
        target = max(0, min(blank_count, target))
        row_counts.append(target)

    for j in range(C):
        blank_count = sum(1 for i in range(R) if grid[i][j] == '.')
        base = blank_count * critical_ratio
        offset = int(random.uniform(-noise, noise) * blank_count)
        target = int(round(base)) + offset
        target = max(0, min(blank_count, target))
        col_counts.append(target)

    return R, C, row_counts, col_counts, grid

def main():
    # Usage: script.py R C tree_prob critical_ratio noise seed output_file
    if len(sys.argv) < 7:
        print("Usage: {} R C tree_prob critical_ratio noise seed output_file".format(sys.argv[0]))
        sys.exit(1)

    R = int(sys.argv[1])
    C = int(sys.argv[2])
    tree_prob = float(sys.argv[3])
    critical_ratio = float(sys.argv[4])
    noise = float(sys.argv[5])
    seed = int(sys.argv[6]) if sys.argv[6] != "None" else None
    output_file = 'hard_inputs/' + sys.argv[7] if len(sys.argv) > 7 else "hard_inputs/input.txt"

    R, C, row_counts, col_counts, grid = generate_input(R, C, tree_prob, critical_ratio, noise, seed)

    with open(output_file, "w") as f:
        # First line: R and C
        f.write(f"{R} {C}\n")
        # Next line: row tent counts
        f.write(" ".join(map(str, row_counts)) + "\n")
        # Next line: column tent counts
        f.write(" ".join(map(str, col_counts)) + "\n")
        # Next R lines: grid rows
        for row in grid:
            f.write("".join(row) + "\n")

if __name__ == '__main__':
    main()
