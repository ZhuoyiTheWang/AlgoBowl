package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal" // added for signal handling
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall" // added for syscall signals
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

	// Bookkeeping for construction: current tent counts for rows and columns.
	rowCounts []int
	colCounts []int
}

// Global ACO parameters.
const (
	numAnts     = 50
	numIters    = 100
	alpha       = 1.0 // pheromone importance
	beta        = 1.5 // heuristic importance
	evaporation = 0.3 // pheromone evaporation rate
	Q           = 316 // pheromone deposit factor
)

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

// parseInput parses the problem input.
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
	if len(rowTargets) != R {
		return nil, fmt.Errorf("row target count mismatch")
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
	if len(colTargets) != C {
		return nil, fmt.Errorf("column target count mismatch")
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

// inBounds returns whether (r,c) is within grid bounds.
func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

// heuristic computes a cell’s heuristic value.
// It rewards cells with adjacent trees and boosts the value if the corresponding row/col are under target.
func (g *Grid) heuristic(r, c int, rowCount, colCount, rowTarget, colTarget int) float64 {
	score := 0.1
	neighbors := [][2]int{{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}
	for _, n := range neighbors {
		nr, nc := n[0], n[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			score += 1.0
		}
	}
	// Boost if row/col are underfilled.
	rowDeficit := float64(rowTarget - rowCount)
	colDeficit := float64(colTarget - colCount)
	if rowDeficit < 0 {
		rowDeficit = 0
	}
	if colDeficit < 0 {
		colDeficit = 0
	}
	score *= (1 + rowDeficit) * (1 + colDeficit)
	return score
}

// assignDirection chooses a pairing direction for a tent at (r,c) based on an adjacent tree.
// It checks the four cardinal directions in order: U, D, L, R. If none are found, it returns 'X'.
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

// abs returns the absolute value of an integer.
func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// globalPairingViolations computes the pairing penalty using a global matching optimizer.
func (g *Grid) globalPairingViolations(sol *Solution) int {
	type pos struct{ r, c int }
	var tents []pos
	var trees []pos

	// Gather all tent positions and tree positions.
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				tents = append(tents, pos{r, c})
			}
			if g.cells[r][c] == 'T' {
				trees = append(trees, pos{r, c})
			}
		}
	}

	// Build an adjacency list: for each tent, list indices of trees that are adjacent (N, S, E, W).
	adj := make([][]int, len(tents))
	for i, t := range tents {
		for j, tr := range trees {
			if (abs(t.r-tr.r) == 1 && t.c == tr.c) || (abs(t.c-tr.c) == 1 && t.r == tr.r) {
				adj[i] = append(adj[i], j)
			}
		}
	}

	// DFS-based bipartite matching.
	matchTree := make([]int, len(trees))
	for i := range matchTree {
		matchTree[i] = -1
	}
	var dfs func(u int, visited []bool) bool
	dfs = func(u int, visited []bool) bool {
		for _, v := range adj[u] {
			if !visited[v] {
				visited[v] = true
				if matchTree[v] == -1 || dfs(matchTree[v], visited) {
					matchTree[v] = u
					return true
				}
			}
		}
		return false
	}
	matchingSize := 0
	for u := 0; u < len(tents); u++ {
		visited := make([]bool, len(trees))
		if dfs(u, visited) {
			matchingSize++
		}
	}

	// Calculate penalty: each unmatched tent and unmatched tree adds a violation.
	penalty := (len(tents) - matchingSize) + (len(trees) - matchingSize)
	return penalty
}

// evaluate computes the total number of violations in a solution.
func (g *Grid) evaluate(sol *Solution) int {
	violations := 0

	// 1. Adjacent tent conflicts (including diagonals).
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				for _, d := range dirs8 {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && sol.placements[nr][nc] {
						violations++
						break // Count at most one violation per tent.
					}
				}
			}
		}
	}

	// 2. Global pairing violations using maximum matching.
	violations += g.globalPairingViolations(sol)

	// 3. Row and column count mismatches.
	for r := 0; r < g.R; r++ {
		violations += int(math.Abs(float64(sol.rowCounts[r] - g.rowTarget[r])))
	}
	for c := 0; c < g.C; c++ {
		violations += int(math.Abs(float64(sol.colCounts[c] - g.colTarget[c])))
	}

	return violations
}

// canPlace returns true if a tent can be placed at (r,c) without violating the non-adjacency rule.
func (sol *Solution) canPlace(g *Grid, r, c int) bool {
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	for _, d := range dirs8 {
		nr, nc := r+d[0], c+d[1]
		if g.inBounds(nr, nc) && sol.placements[nr][nc] {
			return false
		}
	}
	return true
}

// updateCandidates processes a slice of cellCandidate concurrently and returns a new slice
// with updated weights for those candidates that are still eligible.
func updateCandidates(candidates []cellCandidate, sol *Solution, g *Grid, pheromone [][]float64) []cellCandidate {
	numWorkers := runtime.NumCPU()
	chunkSize := (len(candidates) + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	outChan := make(chan cellCandidate, len(candidates))

	for i := 0; i < len(candidates); i += chunkSize {
		end := i + chunkSize
		if end > len(candidates) {
			end = len(candidates)
		}
		wg.Add(1)
		go func(chunk []cellCandidate) {
			defer wg.Done()
			for _, cand := range chunk {
				if sol.canPlace(g, cand.r, cand.c) {
					h := g.heuristic(cand.r, cand.c, sol.rowCounts[cand.r], sol.colCounts[cand.c], g.rowTarget[cand.r], g.colTarget[cand.r])
					cand.weight = math.Pow(pheromone[cand.r][cand.c], alpha) * math.Pow(h, beta)
					outChan <- cand
				}
			}
		}(candidates[i:end])
	}
	wg.Wait()
	close(outChan)

	newCandidates := make([]cellCandidate, 0, len(candidates))
	for cand := range outChan {
		newCandidates = append(newCandidates, cand)
	}
	return newCandidates
}

// cellCandidate represents a blank cell with an associated weight.
type cellCandidate struct {
	r, c   int
	weight float64
}

// constructSolution builds a candidate solution using a greedy randomized approach.
func (g *Grid) constructSolution(pheromone [][]float64, rnd *rand.Rand) *Solution {
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}

	// Build candidate list of all blank cells.
	candidates := []cellCandidate{}
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if g.cells[r][c] == '.' {
				h := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c], g.rowTarget[r], g.colTarget[r])
				w := math.Pow(pheromone[r][c], alpha) * math.Pow(h, beta)
				candidates = append(candidates, cellCandidate{r: r, c: c, weight: w})
			}
		}
	}

	// Greedy randomized selection using roulette wheel.
	for len(candidates) > 0 {
		total := 0.0
		for _, cand := range candidates {
			total += cand.weight
		}
		if total == 0 {
			break
		}
		threshold := rnd.Float64() * total
		sum := 0.0
		var selected cellCandidate
		selectedIndex := -1
		for i, cand := range candidates {
			sum += cand.weight
			if sum >= threshold {
				selected = cand
				selectedIndex = i
				break
			}
		}
		if selectedIndex == -1 {
			break
		}
		// Place a tent if eligible and if row/col are under target.
		if sol.canPlace(g, selected.r, selected.c) &&
			(sol.rowCounts[selected.r] < g.rowTarget[selected.r] || sol.colCounts[selected.c] < g.colTarget[selected.c]) {
			sol.placements[selected.r][selected.c] = true
			sol.rowCounts[selected.r]++
			sol.colCounts[selected.c]++
		}
		// Remove selected candidate.
		candidates = append(candidates[:selectedIndex], candidates[selectedIndex+1:]...)
		// Update remaining candidates concurrently.
		candidates = updateCandidates(candidates, sol, g, pheromone)
	}

	// Repair phase: try to meet any remaining row or column targets.
	// For rows:
	for r := 0; r < g.R; r++ {
		for sol.rowCounts[r] < g.rowTarget[r] {
			bestC := -1
			bestScore := -1.0
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					score := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c], g.rowTarget[r], g.colTarget[c])
					if score > bestScore {
						bestScore = score
						bestC = c
					}
				}
			}
			if bestC == -1 {
				break
			}
			sol.placements[r][bestC] = true
			sol.rowCounts[r]++
			sol.colCounts[bestC]++
		}
	}
	// For columns:
	for c := 0; c < g.C; c++ {
		for sol.colCounts[c] < g.colTarget[c] {
			bestR := -1
			bestScore := -1.0
			for r := 0; r < g.R; r++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					score := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c], g.rowTarget[r], g.colTarget[c])
					if score > bestScore {
						bestScore = score
						bestR = r
					}
				}
			}
			if bestR == -1 {
				break
			}
			sol.placements[bestR][c] = true
			sol.rowCounts[bestR]++
			sol.colCounts[c]++
		}
	}

	sol.violations = g.evaluate(sol)
	return sol
}

// saveBestSolution writes the current best solution to an output file.
func saveBestSolution(bestSol *Solution, grid *Grid, bestViolations int, inputFileName string) {
	outputBuilder := &strings.Builder{}
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
	fmt.Fprintf(outputBuilder, "%d\n", bestViolations)
	fmt.Fprintf(outputBuilder, "%d\n", tentCount)
	for _, t := range placements {
		// Output 1-indexed coordinates.
		fmt.Fprintf(outputBuilder, "%d %d %c\n", t.r+1, t.c+1, t.dir)
	}

	baseName := filepath.Base(inputFileName)
	outputFileName := filepath.Join("outputs", "output_"+baseName)
	os.MkdirAll("outputs", os.ModePerm)
	outFile, err := os.Create(outputFileName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
		return
	}
	defer outFile.Close()

	_, err = outFile.WriteString(outputBuilder.String())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
		return
	}
	fmt.Printf("Solution written to %s\n", outputFileName)
}

func main() {
	// Use all available cores.
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Set up input reader.
	var inputReader *bufio.Reader
	var inputFileName string
	if len(os.Args) > 1 {
		inputFileName = os.Args[1]
		file, err := os.Open(inputFileName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening input file: %v\n", err)
			return
		}
		defer file.Close()
		inputReader = bufio.NewReader(file)
	} else {
		inputReader = bufio.NewReader(os.Stdin)
		inputFileName = "default.txt"
	}

	// Parse input.
	grid, err := parseInput(inputReader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input: %v\n", err)
		return
	}

	// Initialize pheromone matrix.
	pheromone := make([][]float64, grid.R)
	for i := 0; i < grid.R; i++ {
		pheromone[i] = make([]float64, grid.C)
		for j := 0; j < grid.C; j++ {
			if grid.cells[i][j] == '.' {
				pheromone[i][j] = 0.5
			} else {
				pheromone[i][j] = 0.0
			}
		}
	}

	var bestSol *Solution
	bestViolations := math.MaxInt32

	// Global random generator for seeding.
	globalRnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	// Pre-generate seeds for ants.
	seedList := make([]int64, numAnts)
	for i := 0; i < numAnts; i++ {
		seedList[i] = globalRnd.Int63()
	}

	// Set up signal handling for keyboard interruption.
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-signalChan
		fmt.Printf("\nReceived signal %v, saving current best solution...\n", sig)
		if bestSol != nil {
			saveBestSolution(bestSol, grid, bestViolations, inputFileName)
		} else {
			fmt.Println("No solution found yet.")
		}
		os.Exit(1)
	}()

	// Main ACO loop.
	for iter := 0; iter < numIters; iter++ {
		var wg sync.WaitGroup
		solCh := make(chan *Solution, numAnts)
		ants := make([]*Solution, numAnts)

		for k := 0; k < numAnts; k++ {
			wg.Add(1)
			seed := seedList[k]
			go func() {
				defer wg.Done()
				localRnd := rand.New(rand.NewSource(seed))
				sol := grid.constructSolution(pheromone, localRnd)
				solCh <- sol
			}()
		}
		wg.Wait()
		close(solCh)

		i := 0
		for sol := range solCh {
			ants[i] = sol
			i++
		}

		// Find best solution in this iteration.
		sort.Slice(ants, func(i, j int) bool {
			return ants[i].violations < ants[j].violations
		})
		iterBest := ants[0]
		if iterBest.violations < bestViolations {
			bestViolations = iterBest.violations
			bestSol = iterBest
		}

		// Pheromone evaporation.
		for i := 0; i < grid.R; i++ {
			for j := 0; j < grid.C; j++ {
				pheromone[i][j] *= (1 - evaporation)
			}
		}

		// Reinforce using the top 10% (at least one) solutions.
		topK := numAnts / 10
		if topK < 1 {
			topK = 1
		}
		for _, sol := range ants[:topK] {
			deposit := Q / (float64(sol.violations) + 1)
			for r := 0; r < grid.R; r++ {
				for c := 0; c < grid.C; c++ {
					if grid.cells[r][c] == '.' && sol.placements[r][c] {
						pheromone[r][c] += deposit
					}
				}
			}
		}

		// Refresh seeds for next iteration.
		for k := 0; k < numAnts; k++ {
			seedList[k] = globalRnd.Int63()
		}

		fmt.Printf("Iteration %d, best violation: %d\n", iter, iterBest.violations)
	}

	// Save final best solution.
	saveBestSolution(bestSol, grid, bestViolations, inputFileName)
}
