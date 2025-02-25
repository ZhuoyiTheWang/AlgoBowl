package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

type Grid struct {
	R, C      int
	cells     [][]rune
	rowTarget []int
	colTarget []int
}

type Solution struct {
	placements [][]bool   // which cells have tents
	tentMatch  [][][2]int // for each tent cell, which tree (r,c) it claims, or (-1,-1) if unmatched
	rowCounts  []int
	colCounts  []int
	violations int
}

type TentPlacement struct {
	r, c int
	dir  rune
}

// We store local info about a cell candidate
type cellCandidate struct {
	r, c   int
	weight float64
}

// -----------------------------------------------------------------------------
// ACO Parameters
// -----------------------------------------------------------------------------

const (
	numAnts     = 50
	numIters    = 100
	alpha       = 1.0
	beta        = 1.5
	evaporation = 0.3
	Q           = 316

	BATCH_SIZE = 10 // how many tents we place before recalculating candidate weights
)

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// -----------------------------------------------------------------------------
// Utility: parse input
// -----------------------------------------------------------------------------

func readInts(line string) ([]int, error) {
	parts := strings.Fields(line)
	arr := make([]int, len(parts))
	for i, p := range parts {
		num, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		arr[i] = num
	}
	return arr, nil
}

func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

func parseInput(r *bufio.Reader) (*Grid, error) {
	line, err := r.ReadString('\n')
	if err != nil {
		return nil, err
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

	line, err = r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read row targets")
	}
	rowT, err := readInts(line)
	if err != nil {
		return nil, err
	}
	if len(rowT) != R {
		return nil, fmt.Errorf("row target mismatch")
	}

	line, err = r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read col targets")
	}
	colT, err := readInts(line)
	if err != nil {
		return nil, err
	}
	if len(colT) != C {
		return nil, fmt.Errorf("col target mismatch")
	}

	cells := make([][]rune, R)
	for i := 0; i < R; i++ {
		line, err = r.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read grid row %d", i)
		}
		line = strings.TrimSpace(line)
		if len(line) != C {
			return nil, fmt.Errorf("grid row length mismatch at row %d", i)
		}
		cells[i] = []rune(line)
	}

	return &Grid{
		R:         R,
		C:         C,
		cells:     cells,
		rowTarget: rowT,
		colTarget: colT,
	}, nil
}

// -----------------------------------------------------------------------------
// Local "one-tree" approach: each tent can claim at most one tree
// -----------------------------------------------------------------------------

// We'll track for each tree if it's used, so no two tents share the same tree
type localTreeTracker struct {
	used [][]bool
	g    *Grid
}

func newLocalTreeTracker(g *Grid) *localTreeTracker {
	used := make([][]bool, g.R)
	for i := 0; i < g.R; i++ {
		used[i] = make([]bool, g.C)
	}
	return &localTreeTracker{used: used, g: g}
}

// claimFreeTree tries to find any free adjacent tree. If found, claims it and returns (rT, cT).
// If no free tree, return (-1, -1).
func (lt *localTreeTracker) claimFreeTree(r, c int) (int, int) {
	neighbors := [][2]int{{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}
	for _, nb := range neighbors {
		nr, nc := nb[0], nb[1]
		if lt.g.inBounds(nr, nc) && lt.g.cells[nr][nc] == 'T' && !lt.used[nr][nc] {
			// claim
			lt.used[nr][nc] = true
			return nr, nc
		}
	}
	return -1, -1
}

// -----------------------------------------------------------------------------
// Evaluate: adjacency + row/col mismatch + unmatched Tents + unmatched Trees
// -----------------------------------------------------------------------------

func (g *Grid) evaluate(sol *Solution) int {
	// 1. tent adjacency
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	violations := 0
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

	// 2. row/col mismatch
	for r := 0; r < g.R; r++ {
		violations += abs(sol.rowCounts[r] - g.rowTarget[r])
	}
	for c := 0; c < g.C; c++ {
		violations += abs(sol.colCounts[c] - g.colTarget[c])
	}

	// 3. unmatched Tents + unmatched Trees
	// Count how many tents are placed:
	tentCount := 0
	matchedTents := 0
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				tentCount++
				// if it's matched
				if sol.tentMatch[r][c][0] != -1 {
					matchedTents++
				}
			}
		}
	}
	unmatchedTents := tentCount - matchedTents

	// Count total trees, matchedTrees
	totalTrees := 0
	matchedTrees := 0
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if g.cells[r][c] == 'T' {
				totalTrees++
			}
		}
	}
	// Because each matched tent claims exactly 1 tree:
	matchedTrees = matchedTents
	unmatchedTrees := totalTrees - matchedTrees

	violations += unmatchedTents
	violations += unmatchedTrees

	return violations
}

// -----------------------------------------------------------------------------
// Assign direction for final output
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Save best solution
// -----------------------------------------------------------------------------

func saveBestSolution(bestSol *Solution, g *Grid, bestViolations int, inputFileName string) {
	sb := &strings.Builder{}
	tentCount := 0
	var placements []TentPlacement
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if bestSol.placements[r][c] {
				tentCount++
				d := g.assignDirection(r, c)
				placements = append(placements, TentPlacement{r, c, d})
			}
		}
	}
	fmt.Fprintf(sb, "%d\n", bestViolations)
	fmt.Fprintf(sb, "%d\n", tentCount)
	for _, t := range placements {
		fmt.Fprintf(sb, "%d %d %c\n", t.r+1, t.c+1, t.dir)
	}

	baseName := filepath.Base(inputFileName)
	outPath := filepath.Join("outputs", "output_"+baseName)
	_ = os.MkdirAll("outputs", os.ModePerm)
	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output: %v\n", err)
		return
	}
	defer f.Close()
	_, err = f.WriteString(sb.String())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error writing solution: %v\n", err)
	} else {
		fmt.Printf("Solution written to %s\n", outPath)
	}
}

// -----------------------------------------------------------------------------
// Helper: canPlace checks adjacency among tents
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Heuristic for a cell
// -----------------------------------------------------------------------------

func (g *Grid) heuristic(r, c int, rowCount, colCount, rowTarget, colTarget int) float64 {
	score := 0.1
	neighbors := [][2]int{{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}
	for _, nb := range neighbors {
		nr, nc := nb[0], nb[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			score += 1.0
		}
	}
	rowDef := float64(rowTarget - rowCount)
	colDef := float64(colTarget - colCount)
	if rowDef < 0 {
		rowDef = 0
	}
	if colDef < 0 {
		colDef = 0
	}
	score *= (1 + rowDef) * (1 + colDef)
	return score
}

// -----------------------------------------------------------------------------
// ConstructSolution with local partial-matching approach
// -----------------------------------------------------------------------------

func (g *Grid) constructSolution(pheromone [][]float64, rnd *rand.Rand) *Solution {
	// init solution
	sol := &Solution{
		placements: make([][]bool, g.R),
		tentMatch:  make([][][2]int, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
		sol.tentMatch[i] = make([][2]int, g.C)
		for j := 0; j < g.C; j++ {
			sol.tentMatch[i][j] = [2]int{-1, -1}
		}
	}

	// local tree usage
	treeTracker := newLocalTreeTracker(g)

	// build candidate list
	candidates := make([]cellCandidate, 0)
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if g.cells[r][c] == '.' {
				hVal := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c], g.rowTarget[r], g.colTarget[c])
				w := math.Pow(pheromone[r][c], alpha) * math.Pow(hVal, beta)
				candidates = append(candidates, cellCandidate{r, c, w})
			}
		}
	}

	// Batch-based approach
	for {
		placedThisBatch := 0
		for placedThisBatch < BATCH_SIZE && len(candidates) > 0 {
			// Roulette
			totalWeight := 0.0
			for _, cand := range candidates {
				totalWeight += cand.weight
			}
			if totalWeight < 1e-12 {
				break
			}
			threshold := rnd.Float64() * totalWeight
			sum := 0.0
			idx := -1
			for i, cand := range candidates {
				sum += cand.weight
				if sum >= threshold {
					idx = i
					break
				}
			}
			if idx == -1 {
				idx = 0
			}
			chosen := candidates[idx]

			// attempt to place
			if sol.canPlace(g, chosen.r, chosen.c) &&
				sol.rowCounts[chosen.r] < g.rowTarget[chosen.r] &&
				sol.colCounts[chosen.c] < g.colTarget[chosen.c] {

				sol.placements[chosen.r][chosen.c] = true
				sol.rowCounts[chosen.r]++
				sol.colCounts[chosen.c]++

				// try to claim a free adjacent tree
				tr, tc := treeTracker.claimFreeTree(chosen.r, chosen.c)
				if tr != -1 {
					sol.tentMatch[chosen.r][chosen.c] = [2]int{tr, tc}
				} else {
					// remain unmatched -> +1 violation for this tent
				}
				placedThisBatch++
			}

			// remove chosen from candidate list
			candidates[idx] = candidates[len(candidates)-1]
			candidates = candidates[:len(candidates)-1]
		}

		if placedThisBatch == 0 {
			break
		}

		// Recalc weights for the remaining candidates
		for i := 0; i < len(candidates); i++ {
			cand := candidates[i]
			// if adjacency fails, remove
			if !sol.canPlace(g, cand.r, cand.c) {
				candidates[i] = candidates[len(candidates)-1]
				candidates = candidates[:len(candidates)-1]
				i--
				continue
			}
			// if row/col is already at or above target, remove
			if sol.rowCounts[cand.r] >= g.rowTarget[cand.r] ||
				sol.colCounts[cand.c] >= g.colTarget[cand.c] {
				candidates[i] = candidates[len(candidates)-1]
				candidates = candidates[:len(candidates)-1]
				i--
				continue
			}
			// recompute heuristic
			newH := g.heuristic(cand.r, cand.c,
				sol.rowCounts[cand.r], sol.colCounts[cand.c],
				g.rowTarget[cand.r], g.colTarget[cand.c])
			candidates[i].weight = math.Pow(pheromone[cand.r][cand.c], alpha) * math.Pow(newH, beta)
		}
	}

	// final violation
	sol.violations = g.evaluate(sol)
	return sol
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <problemFile>\n", os.Args[0])
		os.Exit(1)
	}
	inputFileName := os.Args[1]

	f, err := os.Open(inputFileName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening problem file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	grid, err := parseInput(reader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing problem: %v\n", err)
		os.Exit(1)
	}

	// init pheromone
	pheromone := make([][]float64, grid.R)
	for r := 0; r < grid.R; r++ {
		pheromone[r] = make([]float64, grid.C)
		for c := 0; c < grid.C; c++ {
			if grid.cells[r][c] == '.' {
				pheromone[r][c] = 0.5
			}
		}
	}

	var bestSol *Solution
	bestViolations := math.MaxInt32

	globalRnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	seedList := make([]int64, numAnts)
	for i := 0; i < numAnts; i++ {
		seedList[i] = globalRnd.Int63()
	}

	// Keyboard interruption => save best
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

	// ACO loop
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

		idx := 0
		for s := range solCh {
			ants[idx] = s
			idx++
		}
		sort.Slice(ants, func(i, j int) bool {
			return ants[i].violations < ants[j].violations
		})
		iterBest := ants[0]
		if iterBest.violations < bestViolations {
			bestViolations = iterBest.violations
			bestSol = iterBest
		}

		// evaporate
		for rr := 0; rr < grid.R; rr++ {
			for cc := 0; cc < grid.C; cc++ {
				pheromone[rr][cc] *= (1 - evaporation)
			}
		}

		// deposit on top 10%
		topK := numAnts / 10
		if topK < 1 {
			topK = 1
		}
		for _, s := range ants[:topK] {
			d := float64(Q) / (float64(s.violations) + 1)
			for rr := 0; rr < grid.R; rr++ {
				for cc := 0; cc < grid.C; cc++ {
					if grid.cells[rr][cc] == '.' && s.placements[rr][cc] {
						pheromone[rr][cc] += d
					}
				}
			}
		}

		// refresh seeds
		for k := 0; k < numAnts; k++ {
			seedList[k] = globalRnd.Int63()
		}

		fmt.Printf("Iteration %d, best violation: %d\n", iter, iterBest.violations)
	}

	if bestSol != nil {
		saveBestSolution(bestSol, grid, bestViolations, inputFileName)
	} else {
		fmt.Println("No solution found.")
	}
}
