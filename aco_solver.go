package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Data structures

// Grid holds the problem data.
type Grid struct {
	R, C      int
	cells     [][]rune // grid of '.' or 'T'
	rowTarget []int
	colTarget []int
}

// TentPlacement represents one tent placement.
type TentPlacement struct {
	r, c int
	dir  rune // U, D, L, R, or X
}

// Solution holds a candidate solution.
type Solution struct {
	placements [][]bool // same dimensions as grid; true means a tent is placed
	violations int
}

// Global ACO parameters
const (
	numAnts     = 50
	numIters    = 100
	alpha       = 1.0  // pheromone importance
	beta        = 2.0  // heuristic importance
	evaporation = 0.1  // pheromone evaporation rate
	Q           = 100.0 // pheromone deposit factor
)

// Helper: read integer slice from a line.
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

// Parse input from standard input.
func parseInput() (*Grid, error) {
	scanner := bufio.NewScanner(os.Stdin)

	// Read first line: R and C.
	if !scanner.Scan() {
		return nil, fmt.Errorf("failed to read first line")
	}
	firstLine := scanner.Text()
	parts := strings.Fields(firstLine)
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

	// Read row targets.
	if !scanner.Scan() {
		return nil, fmt.Errorf("failed to read row targets")
	}
	rowTargets, err := readInts(scanner.Text())
	if err != nil {
		return nil, err
	}
	if len(rowTargets) != R {
		return nil, fmt.Errorf("row target count mismatch")
	}

	// Read column targets.
	if !scanner.Scan() {
		return nil, fmt.Errorf("failed to read column targets")
	}
	colTargets, err := readInts(scanner.Text())
	if err != nil {
		return nil, err
	}
	if len(colTargets) != C {
		return nil, fmt.Errorf("column target count mismatch")
	}

	// Read grid rows.
	cells := make([][]rune, R)
	for i := 0; i < R; i++ {
		if !scanner.Scan() {
			return nil, fmt.Errorf("failed to read grid row %d", i)
		}
		line := scanner.Text()
		if len(line) != C {
			return nil, fmt.Errorf("grid row %d length mismatch", i)
		}
		cells[i] = []rune(line)
	}

	return &Grid{R: R, C: C, cells: cells, rowTarget: rowTargets, colTarget: colTargets}, nil
}

// Returns whether (r,c) is within grid bounds.
func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

// Heuristic value for placing a tent in cell (r,c).
// Here, we simply reward cells that have at least one adjacent tree (N,S,E,W).
func (g *Grid) heuristic(r, c int) float64 {
	neighbors := [][2]int{{r-1, c}, {r+1, c}, {r, c-1}, {r, c+1}}
	score := 0.1
	for _, n := range neighbors {
		nr, nc := n[0], n[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			score += 1.0
		}
	}
	return score
}

// For a tent placed at (r,c), choose a pairing direction: one of U, D, L, R if an adjacent tree exists.
// If none, return 'X'.
func (g *Grid) assignDirection(r, c int) rune {
	dirs := []struct {
		dr, dc int
		dir    rune
	}{
		{-1, 0, 'U'},
		{1, 0, 'D'},
		{0, -1, 'L'},
		{0, 1, 'R'},
	}
	for _, d := range dirs {
		nr, nc := r+d.dr, c+d.dc
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			return d.dir
		}
	}
	return 'X'
}

// Evaluate a candidate solution.
// Computes violations as described: adjacent tents, missing pairings, row/column mismatches.
func (g *Grid) evaluate(sol *Solution) int {
	violations := 0
	// 1. Check adjacent tents (including diagonals)
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1},           {0, 1},
		{1, -1},  {1, 0},  {1, 1},
	}
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				for _, d := range dirs8 {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && sol.placements[nr][nc] {
						violations++
						break // count at most 1 per tent
					}
				}
			}
		}
	}

	// 2. For each tent, if no adjacent tree, add violation.
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				// check N,S,E,W
				found := false
				for _, d := range [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
						found = true
						break
					}
				}
				if !found {
					violations++
				}
			}
		}
	}

	// 3. For each tree, if no adjacent tent, add violation.
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if g.cells[r][c] == 'T' {
				found := false
				for _, d := range [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && sol.placements[nr][nc] {
						found = true
						break
					}
				}
				if !found {
					violations++
				}
			}
		}
	}

	// 4. Row and column count mismatches.
	for r := 0; r < g.R; r++ {
		count := 0
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				count++
			}
		}
		violations += int(math.Abs(float64(count - g.rowTarget[r])))
	}
	for c := 0; c < g.C; c++ {
		count := 0
		for r := 0; r < g.R; r++ {
			if sol.placements[r][c] {
				count++
			}
		}
		violations += int(math.Abs(float64(count - g.colTarget[c])))
	}

	return violations
}

// Ant builds a solution using the pheromone matrix and heuristic info.
func (g *Grid) constructSolution(pheromone [][]float64, rnd *rand.Rand) *Solution {
	// Create empty solution (no tent placements initially)
	placement := make([][]bool, g.R)
	for i := range placement {
		placement[i] = make([]bool, g.C)
	}
	// For each blank cell, decide whether to place a tent.
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			// Only consider blank cells ('.')
			if g.cells[r][c] != '.' {
				continue
			}
			// Compute probability using pheromone and heuristic.
			pher := pheromone[r][c]
			heur := g.heuristic(r, c)
			prob := math.Pow(pher, alpha) * math.Pow(heur, beta)
			// Normalize probability (we use a logistic-like function)
			if prob > 1.0 {
				prob = 1.0
			}
			if rnd.Float64() < prob {
				placement[r][c] = true
			}
		}
	}
	sol := &Solution{placements: placement}
	sol.violations = g.evaluate(sol)
	return sol
}

func main() {
	// Seed the random number generator.
	seed := time.Now().UnixNano()
	rnd := rand.New(rand.NewSource(seed))

	// Parse the input.
	grid, err := parseInput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input: %v\n", err)
		return
	}

	// Initialize pheromone matrix for each cell.
	pheromone := make([][]float64, grid.R)
	for i := 0; i < grid.R; i++ {
		pheromone[i] = make([]float64, grid.C)
		for j := 0; j < grid.C; j++ {
			// For blank cells, initialize with a small positive value.
			if grid.cells[i][j] == '.' {
				pheromone[i][j] = 0.5
			} else {
				pheromone[i][j] = 0.0
			}
		}
	}

	var bestSol *Solution
	bestViolations := math.MaxInt32

	// Main ACO loop.
	for iter := 0; iter < numIters; iter++ {
		var wg sync.WaitGroup
		solCh := make(chan *Solution, numAnts)

		// Launch ants in parallel.
		for k := 0; k < numAnts; k++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				// Each ant gets its own random generator.
				localRnd := rand.New(rand.NewSource(rnd.Int63()))
				sol := grid.constructSolution(pheromone, localRnd)
				solCh <- sol
			}()
		}

		wg.Wait()
		close(solCh)

		// Find best ant solution in this iteration.
		var iterBest *Solution
		iterBestViol := math.MaxInt32
		ants := make([]*Solution, 0, numAnts)
		for sol := range solCh {
			ants = append(ants, sol)
			if sol.violations < iterBestViol {
				iterBestViol = sol.violations
				iterBest = sol
			}
		}
		if iterBestViol < bestViolations {
			bestViolations = iterBestViol
			bestSol = iterBest
		}

		// Update pheromones: evaporation.
		for i := 0; i < grid.R; i++ {
			for j := 0; j < grid.C; j++ {
				pheromone[i][j] *= (1 - evaporation)
			}
		}

		// Reinforce pheromones based on ant solutions.
		// Better solutions deposit more pheromone.
		for _, sol := range ants {
			deposit := Q / (float64(sol.violations) + 1)
			for r := 0; r < grid.R; r++ {
				for c := 0; c < grid.C; c++ {
					if grid.cells[r][c] == '.' && sol.placements[r][c] {
						pheromone[r][c] += deposit
					}
				}
			}
		}

		fmt.Printf("Iteration %d, best violation: %d\n", iter, iterBestViol)
	}

	// Output best solution found.
	// We need to output:
	// First line: total number of violations.
	// Next line: T (number of tents placed).
	// Then T lines: row col direction.
	// We compute direction for each tent using grid.assignDirection.
	tentCount := 0
	placements := []TentPlacement{}
	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if bestSol.placements[r][c] {
				tentCount++
				dir := grid.assignDirection(r, c)
				placements = append(placements, TentPlacement{r: r, c: c, dir: dir})
			}
		}
	}

	// Print output with 1-indexed coordinates.
	fmt.Println(bestViolations)
	fmt.Println(tentCount)
	for _, t := range placements {
		fmt.Printf("%d %d %c\n", t.r+1, t.c+1, t.dir)
	}
}
