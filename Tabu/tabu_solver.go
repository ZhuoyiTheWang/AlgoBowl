package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// -----------------------------
// Data Structures and Types
// -----------------------------

// Grid holds the puzzle data.
type Grid struct {
	R, C      int
	cells     [][]rune // each cell is either '.' (blank) or 'T' (tree)
	rowTarget []int    // required number of tents per row
	colTarget []int    // required number of tents per column
}

// TentPlacement represents one tent placement.
type TentPlacement struct {
	r, c int
	dir  rune // U, D, L, R, or X if no adjacent tree
}

// Solution holds a candidate solution.
type Solution struct {
	placements [][]bool // true means a tent is placed
	violations int

	// Bookkeeping: count of tents per row and column.
	rowCounts []int
	colCounts []int
}

// Move represents a candidate change (either adding or removing a tent).
type Move struct {
	r, c int
	add  bool // true for add, false for remove
}

// CandidateResult holds a candidate move and its computed violation score.
type CandidateResult struct {
	move Move
	val  int
}

// -----------------------------
// Global Variables for Interrupt Handling
// -----------------------------

var globalBestSol *Solution
var globalBestMutex sync.Mutex

// -----------------------------
// Helper Functions
// -----------------------------

// readInts parses a line of space-separated integers.
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

// parseInput reads the puzzle input from a buffered reader.
func parseInput(r *bufio.Reader) (*Grid, error) {
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

// inBounds returns true if (r, c) is within the grid.
func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

// heuristic computes a desirability score for placing a tent at (r, c).
// It rewards cells adjacent to trees and boosts the score if the row/col is underfilled.
func (g *Grid) heuristic(r, c int, rowCount, colCount, rowTarget, colTarget int) float64 {
	score := 0.1
	neighbors := [][2]int{{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}
	for _, n := range neighbors {
		nr, nc := n[0], n[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			score += 1.0
		}
	}
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

// assignDirection returns the first direction (U, D, L, R) where an adjacent tree is found,
// or 'X' if no tree is adjacent.
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
// It builds a bipartite graph between tents and trees and computes the maximum matching.
// The penalty is the sum of unmatched tents and unmatched trees.
func (g *Grid) globalPairingViolations(sol *Solution) int {
	type pos struct{ r, c int }
	var tents []pos
	var trees []pos
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
	adj := make([][]int, len(tents))
	for i, t := range tents {
		for j, tr := range trees {
			if (abs(t.r-tr.r) == 1 && t.c == tr.c) || (abs(t.c-tr.c) == 1 && t.r == tr.r) {
				adj[i] = append(adj[i], j)
			}
		}
	}
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
	penalty := (len(tents) - matchingSize) + (len(trees) - matchingSize)
	return penalty
}

// evaluate computes the overall violation score of a solution.
func (g *Grid) evaluate(sol *Solution) int {
	violations := 0
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	// Adjacent tent conflicts.
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				for _, d := range dirs8 {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && sol.placements[nr][nc] {
						violations++
						break
					}
				}
			}
		}
	}
	violations += g.globalPairingViolations(sol)
	// Row and column mismatches.
	for r := 0; r < g.R; r++ {
		violations += int(math.Abs(float64(sol.rowCounts[r] - g.rowTarget[r])))
	}
	for c := 0; c < g.C; c++ {
		violations += int(math.Abs(float64(sol.colCounts[c] - g.colTarget[c])))
	}
	return violations
}

// canPlace returns true if a tent can be placed at (r, c) without violating non-adjacency.
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

// cloneSolution creates a deep copy of a solution.
func cloneSolution(sol *Solution) *Solution {
	newSol := &Solution{
		placements: make([][]bool, len(sol.placements)),
		rowCounts:  make([]int, len(sol.rowCounts)),
		colCounts:  make([]int, len(sol.colCounts)),
		violations: sol.violations,
	}
	for i := range sol.placements {
		newSol.placements[i] = make([]bool, len(sol.placements[i]))
		copy(newSol.placements[i], sol.placements[i])
	}
	copy(newSol.rowCounts, sol.rowCounts)
	copy(newSol.colCounts, sol.colCounts)
	return newSol
}

// applyMove applies a move (adding or removing a tent) to a solution.
func applyMove(sol *Solution, g *Grid, move Move) *Solution {
	newSol := cloneSolution(sol)
	if move.add {
		if g.cells[move.r][move.c] == '.' && !newSol.placements[move.r][move.c] {
			newSol.placements[move.r][move.c] = true
			newSol.rowCounts[move.r]++
			newSol.colCounts[move.c]++
		}
	} else {
		if newSol.placements[move.r][move.c] {
			newSol.placements[move.r][move.c] = false
			newSol.rowCounts[move.r]--
			newSol.colCounts[move.c]--
		}
	}
	newSol.violations = g.evaluate(newSol)
	return newSol
}

// -----------------------------
// Functions for Loading an Initial Solution
// -----------------------------

// loadSolution reads a solution from a file. The file format is assumed to be:
// line 1: violation count (ignored)
// line 2: tent count (ignored)
// subsequent lines: "row col dir" with 1-indexed coordinates.
func loadSolution(filename string, g *Grid) (*Solution, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}
	// Skip first two lines.
	if !scanner.Scan() {
		return nil, fmt.Errorf("failed to read violation line")
	}
	if !scanner.Scan() {
		return nil, fmt.Errorf("failed to read tent count line")
	}
	// Read placements.
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}
		r, err1 := strconv.Atoi(parts[0])
		c, err2 := strconv.Atoi(parts[1])
		if err1 != nil || err2 != nil {
			continue
		}
		// Convert from 1-indexed to 0-indexed.
		r--
		c--
		if r >= 0 && r < g.R && c >= 0 && c < g.C {
			sol.placements[r][c] = true
			sol.rowCounts[r]++
			sol.colCounts[c]++
		}
	}
	sol.violations = g.evaluate(sol)
	return sol, nil
}

// -----------------------------
// Tabu Search with Fixed Worker Pool and Verbose Output
// -----------------------------

// constructTabuSolution uses Tabu Search to improve a candidate solution.
// It accepts an initial solution (which can be loaded from file) and then optimizes it.
// In this version, we use a fixed worker pool (with 8 workers) to evaluate candidate moves.
func constructTabuSolution(g *Grid, initSol *Solution, maxIter int, tabuTenure int, neighborhoodSize int, rnd *rand.Rand) *Solution {
	current := cloneSolution(initSol)
	best := cloneSolution(current)
	bestVal := current.violations

	globalBestMutex.Lock()
	globalBestSol = cloneSolution(best)
	globalBestMutex.Unlock()

	fmt.Printf("Initial solution violations: %d\n", bestVal)

	tabuList := make(map[Move]int)
	var mu sync.Mutex

	numWorkers := 8

	for iter := 0; iter < maxIter; iter++ {
		// Remove expired moves.
		for m, exp := range tabuList {
			if exp <= iter {
				delete(tabuList, m)
			}
		}

		// Generate candidate moves by randomly sampling neighborhoodSize moves.
		var candidateMoves []Move
		for i := 0; i < neighborhoodSize; i++ {
			r := rnd.Intn(g.R)
			c := rnd.Intn(g.C)
			if g.cells[r][c] != '.' {
				continue
			}
			var m Move
			if current.placements[r][c] {
				m = Move{r: r, c: c, add: false}
			} else {
				m = Move{r: r, c: c, add: true}
			}
			candidateMoves = append(candidateMoves, m)
		}
		fmt.Printf("Iteration %d: Generated %d candidate moves.\n", iter, len(candidateMoves))

		// Create channels for candidate moves and their results.
		candChan := make(chan Move, len(candidateMoves))
		resultChan := make(chan CandidateResult, len(candidateMoves))

		// Worker function: processes candidate moves from candChan.
		worker := func() {
			for m := range candChan {
				newSol := applyMove(current, g, m)
				resultChan <- CandidateResult{move: m, val: newSol.violations}
			}
		}

		// Start fixed worker pool.
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				worker()
			}()
		}

		// Feed candidate moves into the channel.
		for _, m := range candidateMoves {
			candChan <- m
		}
		close(candChan)
		wg.Wait()
		close(resultChan)

		// Choose the best candidate move from results.
		bestMoveFound := false
		bestMoveVal := math.MaxInt32
		var bestMove Move
		for res := range resultChan {
			// If move is tabu and doesn't improve best solution, skip it.
			mu.Lock()
			if _, isTabu := tabuList[res.move]; isTabu && res.val >= bestVal {
				mu.Unlock()
				continue
			}
			if res.val < bestMoveVal {
				bestMoveVal = res.val
				bestMove = res.move
				bestMoveFound = true
			}
			mu.Unlock()
		}

		if !bestMoveFound {
			fmt.Printf("Iteration %d: No acceptable move found. Terminating search.\n", iter)
			break
		}

		moveType := "Remove"
		if bestMove.add {
			moveType = "Add"
		}
		fmt.Printf("Iteration %d: Selected move %s at (%d, %d) leading to violation %d\n",
			iter, moveType, bestMove.r, bestMove.c, bestMoveVal)

		current = applyMove(current, g, bestMove)
		tabuList[bestMove] = iter + tabuTenure

		if current.violations < bestVal {
			bestVal = current.violations
			best = cloneSolution(current)
			fmt.Printf("Iteration %d: New best solution found with violations: %d\n", iter, bestVal)
			globalBestMutex.Lock()
			globalBestSol = cloneSolution(best)
			globalBestMutex.Unlock()
		}
		fmt.Printf("Iteration %d complete: Current violations: %d, Best so far: %d\n", iter, current.violations, bestVal)
		if bestVal == 0 {
			fmt.Println("Zero violations achieved. Terminating search.")
			break
		}
	}

	fmt.Println("Tabu search complete.")
	return best
}

// -----------------------------
// Heuristic Solution Construction (Initial)
// -----------------------------

// constructHeuristicSolution builds a candidate solution using a deterministic greedy heuristic.
func (g *Grid) constructHeuristicSolution() *Solution {
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}
	// Greedy phase.
	for {
		bestR, bestC := 0, 0
		bestScore := -1.0
		found := false
		for r := 0; r < g.R; r++ {
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					if sol.rowCounts[r] < g.rowTarget[r] || sol.colCounts[c] < g.colTarget[c] {
						score := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c], g.rowTarget[r], g.colTarget[c])
						if score > bestScore {
							bestScore = score
							bestR = r
							bestC = c
							found = true
						}
					}
				}
			}
		}
		if !found {
			break
		}
		sol.placements[bestR][bestC] = true
		sol.rowCounts[bestR]++
		sol.colCounts[bestC]++
	}
	// Repair phase for rows.
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
	// Repair phase for columns.
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

// -----------------------------
// Helper to Save the Best Solution to File
// -----------------------------

func saveBestSolution(bestSol *Solution, grid *Grid, inputFileName string) {
	outputBuilder := &strings.Builder{}
	tentCount := 0
	var placements []TentPlacement
	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if bestSol.placements[r][c] {
				tentCount++
				dir := grid.assignDirection(r, c)
				placements = append(placements, TentPlacement{r: r, c: c, dir: dir})
			}
		}
	}
	fmt.Fprintf(outputBuilder, "%d\n", bestSol.violations)
	fmt.Fprintf(outputBuilder, "%d\n", tentCount)
	for _, t := range placements {
		fmt.Fprintf(outputBuilder, "%d %d %c\n", t.r+1, t.c+1, t.dir)
	}

	baseName := filepath.Base(inputFileName)
	outputFileName := filepath.Join("outputs", "output_tabu_"+baseName)
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

// -----------------------------
// Main Function
// -----------------------------

func main() {
	// Use flag package to allow an initial solution file.
	initSolutionFile := flag.String("init", "", "Path to an initial solution file to optimize")
	flag.Parse()

	// Set up input reading.
	var inputReader *bufio.Reader
	var inputFileName string
	if flag.NArg() > 0 {
		inputFileName = flag.Arg(0)
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

	// Parse the grid.
	grid, err := parseInput(inputReader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input: %v\n", err)
		return
	}

	// Initialize random generator.
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set up signal handling for interruption.
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-signalChan
		fmt.Printf("\nReceived signal %v, saving current best solution...\n", sig)
		globalBestMutex.Lock()
		defer globalBestMutex.Unlock()
		if globalBestSol != nil {
			saveBestSolution(globalBestSol, grid, inputFileName)
		} else {
			fmt.Println("No solution found yet.")
		}
		os.Exit(1)
	}()

	// Determine initial solution.
	var initSol *Solution
	if *initSolutionFile != "" {
		fmt.Printf("Loading initial solution from %s\n", *initSolutionFile)
		sol, err := loadSolution(*initSolutionFile, grid)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading initial solution: %v\n", err)
			return
		}
		initSol = sol
	} else {
		fmt.Println("No initial solution provided; using heuristic solution.")
		initSol = grid.constructHeuristicSolution()
	}

	// Run Tabu Search using the fixed worker pool.
	maxIter := 1000         // maximum iterations.
	tabuTenure := 50        // iterations a move remains tabu.
	neighborhoodSize := 100 // candidate moves per iteration.
	bestSol := constructTabuSolution(grid, initSol, maxIter, tabuTenure, neighborhoodSize, rnd)

	// Prepare final output.
	tentCount := 0
	var placements []TentPlacement
	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if bestSol.placements[r][c] {
				tentCount++
				dir := grid.assignDirection(r, c)
				placements = append(placements, TentPlacement{r: r, c: c, dir: dir})
			}
		}
	}
	fmt.Printf("Best solution violations: %d\n", bestSol.violations)
	fmt.Printf("Tent count: %d\n", tentCount)

	// Write final output to file.
	baseName := filepath.Base(inputFileName)
	outputFileName := filepath.Join("outputs", "output_tabu_"+baseName)
	os.MkdirAll("outputs", os.ModePerm)
	outFile, err := os.Create(outputFileName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
		return
	}
	defer outFile.Close()

	fmt.Fprintf(outFile, "%d\n", bestSol.violations)
	fmt.Fprintf(outFile, "%d\n", tentCount)
	for _, t := range placements {
		fmt.Fprintf(outFile, "%d %d %c\n", t.r+1, t.c+1, t.dir)
	}

	fmt.Printf("Solution written to %s\n", outputFileName)
}
