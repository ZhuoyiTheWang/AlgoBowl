import random
import sys

def generate_input(R, C, tree_prob, seed=None):
    if seed is not None:
        random.seed(seed)
    # Generate the grid: randomly assign a tree or a blank to each cell.
    grid = []
    for i in range(R):
        row = []
        for j in range(C):
            cell = 'T' if random.random() < tree_prob else '.'
            row.append(cell)
        grid.append(row)

    # For each row, count the number of blank cells and choose a random tent count (0 to blank_count)
    row_counts = []
    for row in grid:
        count = random.randint(0, 2 * C)
        row_counts.append(count)

    # For each column, count the blank cells and choose a random tent count (0 to blank_count)
    col_counts = []
    for j in range(C):
        count = random.randint(0, 2 * R)
        col_counts.append(count)

    return R, C, row_counts, col_counts, grid

def main():
    # Usage: script.py R C [tree_prob] [seed] [output_file]
    if len(sys.argv) < 4:
        print("Usage: {} R C [tree_prob] [seed] output_file".format(sys.argv[0]))
        sys.exit(1)

    R = int(sys.argv[1])
    C = int(sys.argv[2])
    tree_prob = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    # Seed is optional; if not provided, use None for randomness.
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    # The last argument is the output filename
    output_file = sys.argv[5] if len(sys.argv) > 5 else "input.txt"

    R, C, row_counts, col_counts, grid = generate_input(R, C, tree_prob, seed)

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
