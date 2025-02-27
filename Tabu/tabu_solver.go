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
	cells     [][]rune
	rowTarget []int
	colTarget []int

	// Precomputed positions of '.' and 'T'
	potentialTents []pos
	treePositions  []pos

	// adjacencyList[tentIndex] = slice of all treeIndices
	// that are orthogonally adjacent to that tent cell.
	adjacencyList [][]int
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

// Global best solution references
var globalBestSol *Solution
var globalBestMutex sync.Mutex

// ---------------------------------------------------------------
// Parsing
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

	g := &Grid{
		R:         R,
		C:         C,
		cells:     cells,
		rowTarget: rowT,
		colTarget: colT,
	}

	// Collect potential tents and tree positions
	for rr := 0; rr < R; rr++ {
		for cc := 0; cc < C; cc++ {
			switch g.cells[rr][cc] {
			case 'T':
				g.treePositions = append(g.treePositions, pos{rr, cc})
			case '.':
				g.potentialTents = append(g.potentialTents, pos{rr, cc})
			}
		}
	}

	// Build adjacency once for bipartite matching
	g.adjacencyList = make([][]int, len(g.potentialTents))
	for i, tentPos := range g.potentialTents {
		for j, treePos := range g.treePositions {
			// If orth-adj, record adjacency
			if (abs(tentPos.r-treePos.r) == 1 && tentPos.c == treePos.c) ||
				(abs(tentPos.c-treePos.c) == 1 && tentPos.r == treePos.r) {
				g.adjacencyList[i] = append(g.adjacencyList[i], j)
			}
		}
	}

	return g, nil
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

// ---------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------

// We break out the adjacency checking for tents alone.
func (g *Grid) countTentAdjViolations(sol *Solution) int {
	dirs8 := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	viol := 0

	// Check only active tent positions
	for _, pt := range g.potentialTents {
		r, c := pt.r, pt.c
		if !sol.placements[r][c] {
			continue
		}
		// Check if any of the 8 neighbors is also a tent
		for _, d := range dirs8 {
			nr, nc := r+d[0], c+d[1]
			if g.inBounds(nr, nc) && sol.placements[nr][nc] {
				viol++
				break
			}
		}
	}
	return viol
}

// globalPairingViolations uses the precomputed adjacencyList
func (g *Grid) globalPairingViolations(sol *Solution) int {
	// Gather which tent indices in g.potentialTents are active
	activeTentIndices := make([]int, 0, len(g.potentialTents))
	for i, pt := range g.potentialTents {
		if sol.placements[pt.r][pt.c] {
			activeTentIndices = append(activeTentIndices, i)
		}
	}
	tentCount := len(activeTentIndices)
	treeCount := len(g.treePositions)

	// Prepare match array: matchTree[j] = index (in activeTentIndices) of the tent matched to tree j, or -1
	matchTree := make([]int, treeCount)
	for i := range matchTree {
		matchTree[i] = -1
	}

	// Standard DFS-based bipartite matching
	var dfs func(u int, visited []bool) bool
	dfs = func(u int, visited []bool) bool {
		tentIdx := activeTentIndices[u] // index in adjacencyList
		for _, treeIdx := range g.adjacencyList[tentIdx] {
			if !visited[treeIdx] {
				visited[treeIdx] = true
				if matchTree[treeIdx] == -1 || dfs(matchTree[treeIdx], visited) {
					matchTree[treeIdx] = u
					return true
				}
			}
		}
		return false
	}

	matchingSize := 0
	for u := 0; u < tentCount; u++ {
		visited := make([]bool, treeCount)
		if dfs(u, visited) {
			matchingSize++
		}
	}

	// penalty = (#activeTents - matchingSize) + (#trees - matchingSize)
	return (tentCount - matchingSize) + (treeCount - matchingSize)
}

func (g *Grid) evaluate(sol *Solution) int {
	adjacencyViol := g.countTentAdjViolations(sol)
	pairingViol := g.globalPairingViolations(sol)

	// row/col mismatches
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
// Cloning & Moves
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

// ---------------------------------------------------------------
// Tabu Search
// ---------------------------------------------------------------
func constructTabuSolution(g *Grid, initSol *Solution, maxIter, tabuTenure, neighborSize int, rnd *rand.Rand) *Solution {
	current := cloneSolution(initSol)
	best := cloneSolution(current)
	bestVal := current.violations

	globalBestMutex.Lock()
	globalBestSol = cloneSolution(best)
	globalBestMutex.Unlock()

	fmt.Printf("Initial solution violations: %d\n", bestVal)

	tabuList := make(map[Move]int)

	// Preallocate candidate moves slice
	candidateMoves := make([]Move, 0, neighborSize)

	// Create persistent channels for moves and results
	candChan := make(chan Move, neighborSize)
	resChan := make(chan CandidateResult, neighborSize)

	numWorkers := 8
	var wg sync.WaitGroup

	// Start persistent worker pool
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for m := range candChan {
				ns := applyMove(current, g, m)
				resChan <- CandidateResult{m, ns.violations}
			}
		}()
	}

	for iter := 0; iter < maxIter; iter++ {
		// Remove expired entries in tabu list
		for m, exp := range tabuList {
			if exp <= iter {
				delete(tabuList, m)
			}
		}

		// Generate random candidate moves
		candidateMoves = candidateMoves[:0]
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

		fmt.Printf("Iteration %d: Generated %d candidate moves.\n", iter, len(candidateMoves))

		// Send them to worker pool
		for _, cm := range candidateMoves {
			candChan <- cm
		}

		// Collect results
		bestMoveFound := false
		bestMoveVal := math.MaxInt32
		var bestMove Move

		for i := 0; i < len(candidateMoves); i++ {
			cr := <-resChan
			if _, isTabu := tabuList[cr.move]; isTabu && cr.val >= bestVal {
				// Skip tabooed moves that don't improve
				continue
			}
			if cr.val < bestMoveVal {
				bestMoveVal = cr.val
				bestMove = cr.move
				bestMoveFound = true
			}
		}

		if !bestMoveFound {
			fmt.Printf("Iteration %d: No acceptable move found, terminating.\n", iter)
			break
		}

		moveStr := "Remove"
		if bestMove.add {
			moveStr = "Add"
		}
		fmt.Printf("Iteration %d: Chosen move %s at (%d,%d) => viol %d\n",
			iter, moveStr, bestMove.r, bestMove.c, bestMoveVal)

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

	// Terminate workers
	close(candChan)
	wg.Wait()
	close(resChan)

	fmt.Println("Tabu search complete.")
	return best
}

// ---------------------------------------------------------------
// Loading an Initial Solution
// ---------------------------------------------------------------
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

	// Skip the first 2 lines
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
// Saving the Final Solution
// ---------------------------------------------------------------
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

// We'll reuse the bipartite logic to figure out which tree each tent is matched to.
// This is effectively "computeBipartiteMatchingDetailed" from earlier, but updated
// to use our precomputed adjacency and the "active" tent indices approach.
func (g *Grid) computeBipartiteMatchingDetailed(sol *Solution) (matchTent []int, tents []pos, trees []pos) {
	// Build up the list of actual tent positions
	var activeTentIndices []int
	for i, pt := range g.potentialTents {
		if sol.placements[pt.r][pt.c] {
			activeTentIndices = append(activeTentIndices, i)
			tents = append(tents, pt)
		}
	}
	// All tree positions
	trees = g.treePositions

	treeCount := len(trees)
	matchTree := make([]int, treeCount)
	for i := range matchTree {
		matchTree[i] = -1
	}

	// DFS as usual
	var dfs func(u int, visited []bool) bool
	dfs = func(u int, visited []bool) bool {
		tentIdx := activeTentIndices[u]
		for _, treeIdx := range g.adjacencyList[tentIdx] {
			if !visited[treeIdx] {
				visited[treeIdx] = true
				if matchTree[treeIdx] == -1 || dfs(matchTree[treeIdx], visited) {
					matchTree[treeIdx] = u
					return true
				}
			}
		}
		return false
	}

	for u := 0; u < len(activeTentIndices); u++ {
		visited := make([]bool, treeCount)
		_ = dfs(u, visited)
	}

	// matchTent[i] = index of tree matched to the i-th tent (in the "tents" array),
	// or -1 if none
	matchTent = make([]int, len(activeTentIndices))
	for i := range matchTent {
		matchTent[i] = -1
	}
	// If matchTree[treeIdx] = u, then the tent with local index u is matched to tree treeIdx
	// That tent's global index is activeTentIndices[u].
	// We want to record matchTent[u] = treeIdx
	for treeIdx, u := range matchTree {
		if u != -1 {
			matchTent[u] = treeIdx
		}
	}
	return matchTent, tents, trees
}

func saveBestSolution(bestSol *Solution, g *Grid, inputFileName string) {
	matchTent, tents, trees := g.computeBipartiteMatchingDetailed(bestSol)

	out := &strings.Builder{}

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

	for i, p := range tents {
		treeIdx := matchTent[i]
		var dir rune
		if treeIdx == -1 {
			dir = 'X'
		} else {
			treeR, treeC := trees[treeIdx].r, trees[treeIdx].c
			dir = computeDirection(p.r, p.c, treeR, treeC)
		}
		fmt.Fprintf(out, "%d %d %c\n", p.r+1, p.c+1, dir)
	}

	baseName := filepath.Base(inputFileName)
	outputFileName := filepath.Join("tabu", "output_"+baseName)
	os.MkdirAll("tabu", os.ModePerm)

	f, err := os.Create(outputFileName)
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
	fmt.Printf("Solution written to %s\n", outputFileName)
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

	// Graceful interruption
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

	// Require an initial solution
	if *initSolutionPath == "" {
		fmt.Fprintln(os.Stderr, "No init provided. You must provide -init. Exiting.")
		return
	}
	fmt.Printf("Loading init from %s\n", *initSolutionPath)
	initSol, err := loadSolution(*initSolutionPath, g)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading init solution: %v\n", err)
		return
	}
	fmt.Printf("Initial loaded => %d violations\n", initSol.violations)

	// Tabu search parameters
	maxIter := 2500
	tabuTenure := 50
	neighborhoodSize := 400

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	bestSol := constructTabuSolution(g, initSol, maxIter, tabuTenure, neighborhoodSize, rnd)

	fmt.Printf("Best solution => %d violations\n", bestSol.violations)
	if bestSol.violations < initSol.violations {
		difference := initSol.violations - bestSol.violations
		fmt.Printf("Reduction of %d violations\n", difference)
	} else {
		fmt.Printf("No change in violation count\n")
	}
	saveBestSolution(bestSol, g, inputFileName)
}
