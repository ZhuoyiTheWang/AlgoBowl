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

// ---------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------
type pos struct{ r, c int }

type Grid struct {
	R, C      int
	cells     [][]rune // '.' or 'T'
	rowTarget []int
	colTarget []int
}

type TentPlacement struct {
	r, c int
	dir  rune
}

type Solution struct {
	placements [][]bool
	violations int

	// row/col counts
	rowCounts []int
	colCounts []int
}

type Move struct {
	r, c int
	add  bool
}

type CandidateResult struct {
	move Move
	val  int
}

// For graceful interruption, storing the global best
var globalBestSol *Solution
var globalBestMutex sync.Mutex

// ---------------------------------------------------------------
// Parsing and helpers
// ---------------------------------------------------------------

func readInts(line string) ([]int, error) {
	parts := strings.Fields(line)
	nums := make([]int, len(parts))
	for i, p := range parts {
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		nums[i] = v
	}
	return nums, nil
}

func parseInput(r *bufio.Reader) (*Grid, error) {
	line, err := r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("fail read first line: %v", err)
	}
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return nil, fmt.Errorf("not enough data in first line")
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
		return nil, fmt.Errorf("fail row targets: %v", err)
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
		return nil, fmt.Errorf("fail col targets: %v", err)
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
			return nil, fmt.Errorf("fail read row %d: %v", i, err)
		}
		line = strings.TrimSpace(line)
		if len(line) != C {
			return nil, fmt.Errorf("grid row mismatch at %d", i)
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

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (g *Grid) inBounds(r, c int) bool {
	return (r >= 0 && r < g.R && c >= 0 && c < g.C)
}

// adjacency check for solution
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

// direction for final output
func (g *Grid) assignDirection(r, c int) rune {
	// naive approach: just pick the first cardinal direction that is T
	dirs := []struct {
		dr, dc int
		label  rune
	}{
		{-1, 0, 'U'},
		{1, 0, 'D'},
		{0, -1, 'L'},
		{0, 1, 'R'},
	}
	for _, d := range dirs {
		nr, nc := r+d.dr, c+d.dc
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			return d.label
		}
	}
	return 'X'
}

// ---------------------------------------------------------------
// BFS-based bipartite matching used for violation
// ---------------------------------------------------------------

func (g *Grid) globalPairingViolations(sol *Solution) int {
	// gather tents and trees

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

	// adjacency list
	adj := make([][]int, len(tents))
	for i, t := range tents {
		for j, tr := range trees {
			// if up/down/left/right
			if (abs(t.r-tr.r) == 1 && t.c == tr.c) ||
				(abs(t.c-tr.c) == 1 && t.r == tr.r) {
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

	// unmatched tents + unmatched trees
	penalty := (len(tents) - matchingSize) + (len(trees) - matchingSize)
	return penalty
}

// same bipartite approach, but we return the array that says which tent matched which tree
func (g *Grid) computeBipartiteMatchingDetailed(sol *Solution) (matchTent []int, tents []pos, trees []pos) {
	// gather
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				tents = append(tents, pos{r, c})
			}
		}
	}
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if g.cells[r][c] == 'T' {
				trees = append(trees, pos{r, c})
			}
		}
	}

	// adjacency
	adj := make([][]int, len(tents))
	for i, t := range tents {
		for j, tr := range trees {
			if (abs(t.r-tr.r) == 1 && t.c == tr.c) ||
				(abs(t.c-tr.c) == 1 && t.r == tr.r) {
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

	// find maximum matching
	for u := 0; u < len(tents); u++ {
		visited := make([]bool, len(trees))
		_ = dfs(u, visited)
	}

	// Now we want matchTent[i] = indexOfTree matched with tent i, or -1
	matchTent = make([]int, len(tents))
	for i := 0; i < len(tents); i++ {
		matchTent[i] = -1
	}
	for j, i := range matchTree {
		if i != -1 {
			matchTent[i] = j
		}
	}
	return matchTent, tents, trees
}

// evaluate function
func (g *Grid) evaluate(sol *Solution) int {
	// adjacency
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	adjacencyViol := 0
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				for _, d := range dirs8 {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && sol.placements[nr][nc] {
						adjacencyViol++
						break
					}
				}
			}
		}
	}

	// bipartite matching
	pairingViol := g.globalPairingViolations(sol)

	// row/col mismatch
	mismatchViol := 0
	for r := 0; r < g.R; r++ {
		mismatchViol += abs(sol.rowCounts[r] - g.rowTarget[r])
	}
	for c := 0; c < g.C; c++ {
		mismatchViol += abs(sol.colCounts[c] - g.colTarget[c])
	}

	return adjacencyViol + pairingViol + mismatchViol
}

// ---------------------------------------------------------------
// Tabu logic
// ---------------------------------------------------------------

func cloneSolution(sol *Solution) *Solution {
	ns := &Solution{
		placements: make([][]bool, len(sol.placements)),
		rowCounts:  make([]int, len(sol.rowCounts)),
		colCounts:  make([]int, len(sol.colCounts)),
		violations: sol.violations,
	}
	for i := 0; i < len(sol.placements); i++ {
		ns.placements[i] = make([]bool, len(sol.placements[i]))
		copy(ns.placements[i], sol.placements[i])
	}
	copy(ns.rowCounts, sol.rowCounts)
	copy(ns.colCounts, sol.colCounts)
	return ns
}

// apply a single move
func applyMove(sol *Solution, g *Grid, move Move) *Solution {
	ns := cloneSolution(sol)
	if move.add {
		if g.cells[move.r][move.c] == '.' && !ns.placements[move.r][move.c] {
			ns.placements[move.r][move.c] = true
			ns.rowCounts[move.r]++
			ns.colCounts[move.c]++
		}
	} else {
		if ns.placements[move.r][move.c] {
			ns.placements[move.r][move.c] = false
			ns.rowCounts[move.r]--
			ns.colCounts[move.c]--
		}
	}
	ns.violations = g.evaluate(ns)
	return ns
}

func constructTabuSolution(g *Grid, initSol *Solution, maxIter int, tabuTenure int, neighborSize int, rnd *rand.Rand) *Solution {
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
		// remove expired
		for m, exp := range tabuList {
			if exp <= iter {
				delete(tabuList, m)
			}
		}

		// build candidate moves
		var candidateMoves []Move
		for i := 0; i < neighborSize; i++ {
			rr := rnd.Intn(g.R)
			cc := rnd.Intn(g.C)
			if g.cells[rr][cc] != '.' {
				continue
			}
			if current.placements[rr][cc] {
				candidateMoves = append(candidateMoves, Move{rr, cc, false})
			} else {
				candidateMoves = append(candidateMoves, Move{rr, cc, true})
			}
		}

		candChan := make(chan Move, len(candidateMoves))
		resChan := make(chan CandidateResult, len(candidateMoves))

		// workers
		worker := func() {
			for m := range candChan {
				ns := applyMove(current, g, m)
				resChan <- CandidateResult{m, ns.violations}
			}
		}

		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				worker()
			}()
		}

		for _, cm := range candidateMoves {
			candChan <- cm
		}
		close(candChan)
		wg.Wait()
		close(resChan)

		bestMoveFound := false
		bestMoveVal := math.MaxInt32
		var bestMove Move

		for r := range resChan {
			mu.Lock()
			if _, isTabu := tabuList[r.move]; isTabu && r.val >= bestVal {
				mu.Unlock()
				continue
			}
			if r.val < bestMoveVal {
				bestMoveVal = r.val
				bestMove = r.move
				bestMoveFound = true
			}
			mu.Unlock()
		}

		if !bestMoveFound {
			fmt.Printf("Iteration %d: No acceptable move found, terminating.\n", iter)
			break
		}

		fmt.Printf("Iteration %d:  viol %d\n", iter, bestMoveVal)

		current = applyMove(current, g, bestMove)
		tabuList[bestMove] = iter + tabuTenure
		if current.violations < bestVal {
			bestVal = current.violations
			best = cloneSolution(current)
			fmt.Printf("Iteration %d: new best => %d violations\n", iter, bestVal)
			globalBestMutex.Lock()
			globalBestSol = cloneSolution(best)
			globalBestMutex.Unlock()
		}
		if bestVal == 0 {
			fmt.Println("Zero violation => done")
			break
		}
	}

	fmt.Println("Tabu search complete.")
	return best
}

// ---------------------------------------------------------------
// Heuristic initial solution (KEPT, but never called if -init is not provided)
// ---------------------------------------------------------------
func (g *Grid) constructHeuristicSolution() *Solution {
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}

	// 1) greedy
	for {
		bestScore := -1.0
		bestR := -1
		bestC := -1
		found := false
		for r := 0; r < g.R; r++ {
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					if sol.rowCounts[r] < g.rowTarget[r] || sol.colCounts[c] < g.colTarget[c] {
						sc := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c],
							g.rowTarget[r], g.colTarget[c])
						if sc > bestScore {
							bestScore = sc
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

	// 2) row fix
	for r := 0; r < g.R; r++ {
		for sol.rowCounts[r] < g.rowTarget[r] {
			bestScore := -1.0
			bestC := -1
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					sc := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c],
						g.rowTarget[r], g.colTarget[c])
					if sc > bestScore {
						bestScore = sc
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
	// 3) col fix
	for c := 0; c < g.C; c++ {
		for sol.colCounts[c] < g.colTarget[c] {
			bestScore := -1.0
			bestR := -1
			for r := 0; r < g.R; r++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					sc := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c],
						g.rowTarget[r], g.colTarget[c])
					if sc > bestScore {
						bestScore = sc
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

func (g *Grid) heuristic(r, c int, rowCount, colCount, rowT, colT int) float64 {
	base := 0.1
	dirs := [][2]int{{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}
	for _, d := range dirs {
		nr, nc := d[0], d[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			base += 1.0
		}
	}
	rowDef := float64(rowT - rowCount)
	colDef := float64(colT - colCount)
	if rowDef < 0 {
		rowDef = 0
	}
	if colDef < 0 {
		colDef = 0
	}
	base *= (1 + rowDef) * (1 + colDef)
	return base
}

// loading from a file for -init
func loadSolution(filename string, g *Grid) (*Solution, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}
	// skip first 2 lines
	if !sc.Scan() {
		return nil, fmt.Errorf("missing line1")
	}
	if !sc.Scan() {
		return nil, fmt.Errorf("missing line2")
	}
	for sc.Scan() {
		line := sc.Text()
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}
		rr, e1 := strconv.Atoi(parts[0])
		cc, e2 := strconv.Atoi(parts[1])
		if e1 != nil || e2 != nil {
			continue
		}
		rr--
		cc--
		if rr >= 0 && rr < g.R && cc >= 0 && cc < g.C {
			sol.placements[rr][cc] = true
			sol.rowCounts[rr]++
			sol.colCounts[cc]++
		}
	}
	sol.violations = g.evaluate(sol)
	return sol, nil
}

// ---------------------------------------------------------------
// Save final solution, but with bipartite matching directions
// ---------------------------------------------------------------

// compute direction from tent to tree, if they're exactly up/down/left/right. Else 'X'.
func computeDirection(tentR, tentC, treeR, treeC int) rune {
	if tentR == treeR {
		if treeC == tentC-1 {
			return 'L'
		}
		if treeC == tentC+1 {
			return 'R'
		}
	}
	if tentC == treeC {
		if treeR == tentR-1 {
			return 'U'
		}
		if treeR == tentR+1 {
			return 'D'
		}
	}
	return 'X'
}

func saveBestSolution(bestSol *Solution, g *Grid, inputFileName string) {
	// We'll do the bipartite matching again, then print directions from that matching
	matchTent, tents, trees := g.computeBipartiteMatchingDetailed(bestSol)

	// build final lines from matched pairs
	out := &strings.Builder{}

	// count how many tents
	tentCount := 0
	for _, row := range bestSol.placements {
		for _, b := range row {
			if b {
				tentCount++
			}
		}
	}
	fmt.Fprintf(out, "%d\n", bestSol.violations)
	fmt.Fprintf(out, "%d\n", tentCount)

	// we want to output them in row-major order
	idx := 0 // index in the 'tents' array
	for i := range tents {
		tentR, tentC := tents[i].r, tents[i].c
		treeIdx := matchTent[i]
		var dir rune
		if treeIdx == -1 {
			dir = 'X'
		} else {
			treeR, treeC := trees[treeIdx].r, trees[treeIdx].c
			dir = computeDirection(tentR, tentC, treeR, treeC)
		}
		// output is 1-based
		fmt.Fprintf(out, "%d %d %c\n", tentR+1, tentC+1, dir)
		idx++
	}

	baseName := filepath.Base(inputFileName)
	outFileName := filepath.Join("tabu", "output_"+baseName)
	os.MkdirAll("tabu", os.ModePerm)

	f, err := os.Create(outFileName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error create output: %v\n", err)
		return
	}
	defer f.Close()

	_, err = f.WriteString(out.String())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error writing: %v\n", err)
		return
	}
	fmt.Printf("Solution written to %s\n", outFileName)
}

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------

func main() {
	initSolutionPath := flag.String("init", "", "Path to an initial solution file.")
	flag.Parse()

	var inputReader *bufio.Reader
	var inputFileName string
	if flag.NArg() > 0 {
		inputFileName = flag.Arg(0)
		fi, err := os.Open(inputFileName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening input: %v\n", err)
			return
		}
		defer fi.Close()
		inputReader = bufio.NewReader(fi)
	} else {
		inputReader = bufio.NewReader(os.Stdin)
		inputFileName = "default.txt"
	}

	g, err := parseInput(inputReader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Parse error: %v\n", err)
		return
	}

	// graceful interrupt
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-signalChan
		fmt.Printf("\nReceived signal %v, saving best...\n", sig)
		globalBestMutex.Lock()
		if globalBestSol != nil {
			saveBestSolution(globalBestSol, g, inputFileName)
		} else {
			fmt.Println("No solution found yet.")
		}
		os.Exit(1)
	}()

	// Must provide warm start: if -init not provided, exit.
	var initSol *Solution
	if *initSolutionPath != "" {
		fmt.Printf("Loading init from %s\n", *initSolutionPath)
		so, err := loadSolution(*initSolutionPath, g)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading init solution: %v\n", err)
			return
		}
		initSol = so
		fmt.Printf("Initial loaded => %d violations\n", initSol.violations)
	} else {
		fmt.Fprintln(os.Stderr, "No init provided. You must provide -init. Exiting.")
		return
	}

	// Tabu search parameters
	maxIter := 2500
	tabuTenure := 200
	neighborhoodSize := 100

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	bestSol := constructTabuSolution(g, initSol, maxIter, tabuTenure, neighborhoodSize, rnd)

	// final output
	fmt.Printf("Best solution => %d violations\n", bestSol.violations)
	if bestSol.violations < initSol.violations {
		difference := initSol.violations - bestSol.violations
		fmt.Printf("Reduction of %d violations\n", difference)
	} else {
		fmt.Printf("No change in violation count\n")
	}
	saveBestSolution(bestSol, g, inputFileName)
}
