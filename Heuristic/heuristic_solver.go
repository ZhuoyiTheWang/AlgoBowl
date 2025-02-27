package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// -----------------------------
// DSU (Disjoint Set Union) Data Structure
// -----------------------------

// DSU represents Disjoint Set Union for clustering trees.
type DSU struct {
	Parent, Rank []int
}

// Creates a new DSU instance for `n` elements.
func NewDSU(n int) *DSU {
	dsu := &DSU{
		Parent: make([]int, n),
		Rank:   make([]int, n),
	}
	for i := range dsu.Parent {
		dsu.Parent[i] = i
	}
	return dsu
}

// Find operation with path compression.
func (dsu *DSU) Find(x int) int {
	if dsu.Parent[x] != x {
		dsu.Parent[x] = dsu.Find(dsu.Parent[x]) // Path compression
	}
	return dsu.Parent[x]
}

// Union operation with rank optimization.
func (dsu *DSU) Union(x, y int) {
	rootX, rootY := dsu.Find(x), dsu.Find(y)
	if rootX != rootY {
		if dsu.Rank[rootX] > dsu.Rank[rootY] {
			dsu.Parent[rootY] = rootX
		} else {
			dsu.Parent[rootX] = rootY
			if dsu.Rank[rootX] == dsu.Rank[rootY] {
				dsu.Rank[rootY]++
			}
		}
	}
}

func clusterTrees(grid *Grid) *DSU {
	dsu := NewDSU(grid.R * grid.C)
	directions := []pos{
		{-1, 0}, {1, 0}, {0, -1}, {0, 1},
	}

	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if grid.cells[r][c] == 'T' {
				for _, d := range directions {
					nr, nc := r+d.r, c+d.c
					if grid.inBounds(nr, nc) && grid.cells[nr][nc] == 'T' {
						dsu.Union(index(r, c, grid.C), index(nr, nc, grid.C))
					}
				}
			}
		}
	}

	return dsu
}

func placeTents(grid *Grid, dsu *DSU) *Solution {
	directions := []pos{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	pairedTrees := make(map[int]bool)
	sol := &Solution{
		placements: make([][]bool, grid.R),
		rowCounts:  make([]int, grid.R),
		colCounts:  make([]int, grid.C),
	}

	for i := 0; i < grid.R; i++ {
		sol.placements[i] = make([]bool, grid.C)
	}

	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if grid.cells[r][c] == 'T' {
				root := dsu.Find(index(r, c, grid.C))

				if pairedTrees[root] {
					continue
				}

				for _, d := range directions {
					nr, nc := r+d.r, c+d.c
					if grid.inBounds(nr, nc) && grid.cells[nr][nc] == '.' && sol.canPlace(grid, nr, nc) {
						sol.placements[nr][nc] = true
						sol.rowCounts[nr]++
						sol.colCounts[nc]++
						pairedTrees[root] = true
						break
					}
				}
			}
		}
	}

	sol.violations = grid.evaluate(sol)
	return sol
}

func index(r, c, cols int) int {
	return r*cols + c
}

// -----------------------------
// Grid & Solution Data Structures
// -----------------------------

type pos struct {
	r, c int
}

// Grid holds the problem data.
type Grid struct {
	R, C      int
	cells     [][]rune
	rowTarget []int
	colTarget []int
}

// Solution holds a candidate solution.
type Solution struct {
	placements [][]bool
	violations int
	rowCounts  []int
	colCounts  []int
}

// Determines if placement is valid (prevents adjacent tents)
func (s *Solution) canPlace(grid *Grid, r, c int) bool {
	directions := []pos{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for _, d := range directions {
		nr, nc := r+d.r, c+d.c
		if grid.inBounds(nr, nc) && s.placements[nr][nc] {
			return false
		}
	}
	return true
}

// Evaluates violations in a solution.
func (g *Grid) evaluate(s *Solution) int {
	violations := 0

	// Row violations
	for r := 0; r < g.R; r++ {
		if s.rowCounts[r] != g.rowTarget[r] {
			violations++
		}
	}

	// Column violations
	for c := 0; c < g.C; c++ {
		if s.colCounts[c] != g.colTarget[c] {
			violations++
		}
	}

	return violations
}

// inBounds returns whether (r, c) is within grid bounds.
func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

// -----------------------------
// Parsing Functions
// -----------------------------

// readInts reads a slice of ints from a line.
func readInts(line string) ([]int, error) {
	parts := strings.Fields(line)
	res := make([]int, len(parts))
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		res[i] = n
	}
	return res, nil
}

// parseInput reads and parses the problem input from a file.
func parseInput(r *bufio.Reader) (*Grid, error) {
	// First line: R and C.
	line, err := r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read first line: %v", err)
	}
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid first line")
	}
	R, err := strconv.Atoi(parts[0])
	if err != nil {
		return nil, err
	}
	C, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, err
	}

	// Row targets.
	line, err = r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read row targets: %v", err)
	}
	rowTargets, err := readInts(line)
	if err != nil {
		return nil, err
	}

	// Column targets.
	line, err = r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read column targets: %v", err)
	}
	colTargets, err := readInts(line)
	if err != nil {
		return nil, err
	}

	// Grid rows.
	cells := make([][]rune, R)
	for i := 0; i < R; i++ {
		line, err = r.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read grid row %d: %v", i, err)
		}
		line = strings.TrimSpace(line)
		if len(line) != C {
			return nil, fmt.Errorf("grid row %d length mismatch", i)
		}
		cells[i] = []rune(line)
	}

	return &Grid{R: R, C: C, cells: cells, rowTarget: rowTargets, colTarget: colTargets}, nil
}

// -----------------------------
// Main Function
// -----------------------------

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <input_filename>")
		return
	}

	filename := os.Args[1]
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	grid, err := parseInput(bufio.NewReader(file))
	if err != nil {
		fmt.Println("Error parsing input:", err)
		return
	}

	start := time.Now()
	dsu := clusterTrees(grid)
	sol := placeTents(grid, dsu)
	elapsed := time.Since(start)

	fmt.Printf("DSU solution constructed in %v\n", elapsed)
	fmt.Printf("Total violations: %d\n", sol.violations)

	// Save the solution to an output file
	outputFile := filepath.Join("outputs", "output_"+filepath.Base(filename))
	os.MkdirAll("outputs", os.ModePerm)
	outFile, _ := os.Create(outputFile)
	defer outFile.Close()

	for r := range sol.placements {
		for c := range sol.placements[r] {
			if sol.placements[r][c] {
				fmt.Fprintf(outFile, "%d %d X\n", r+1, c+1)
			}
		}
	}
	fmt.Printf("Solution written to %s\n", outputFile)
}
