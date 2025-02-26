package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// -----------------------------
// Data Structures
// -----------------------------

// pos represents a grid position.
type pos struct {
	r, c int
}

// Grid holds the puzzle data.
type Grid struct {
	R, C      int
	cells     [][]rune // each cell is '.' (blank) or 'T' (tree)
	rowTarget []int    // desired number of tents in each row
	colTarget []int    // desired number of tents in each column
}

// Solution represents a candidate solution.
type Solution struct {
	placements [][]bool // true means a tent is placed at that cell
	rowCounts  []int    // current tent count per row
	colCounts  []int    // current tent count per column
	violations int      // overall violation count (computed using all rules)
}

// -----------------------------
// Parsing Functions
// -----------------------------

// readInts parses a line of space-separated integers.
func readInts(line string) ([]int, error) {
	parts := strings.Fields(line)
	nums := make([]int, len(parts))
	for i, s := range parts {
		n, err := strconv.Atoi(s)
		if err != nil {
			return nil, err
		}
		nums[i] = n
	}
	return nums, nil
}

// parseInput reads the puzzle input from the provided reader.
func parseInput(r *bufio.Reader) (*Grid, error) {
	// First line: number of rows and columns.
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

	// Next line: row targets.
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

	// Next line: column targets.
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

	// Next R lines: grid rows.
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

	return &Grid{
		R:         R,
		C:         C,
		cells:     cells,
		rowTarget: rowTargets,
		colTarget: colTargets,
	}, nil
}

// inBounds returns whether (r,c) is within grid bounds.
func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// -----------------------------
// Overall Evaluation Function
// -----------------------------

// evaluate computes the overall violation score of a solution.
// It adds three kinds of penalties:
// 1. Each tent that has any adjacent (8-neighbor) tent adds 1 violation.
// 2. Using bipartite matching, each unmatched tent or tree adds 1 violation.
// 3. For each row and column, the absolute difference between the tent count and target adds 1 violation per extra/missing tent.
// Note: For the flexible heuristic, we do NOT include the local blocking cost in the final evaluation.
func (g *Grid) evaluate(sol *Solution) int {
	// 1. Adjacency violations.
	adjViol := 0
	dirs8 := [][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				for _, d := range dirs8 {
					nr, nc := r+d[0], c+d[1]
					if g.inBounds(nr, nc) && sol.placements[nr][nc] {
						adjViol++
						break
					}
				}
			}
		}
	}

	// 2. Pairing violations.
	pairViol := g.globalPairingViolations(sol)

	// 3. Row/column mismatches.
	rowMismatch := 0
	for r := 0; r < g.R; r++ {
		rowMismatch += abs(sol.rowCounts[r] - g.rowTarget[r])
	}
	colMismatch := 0
	for c := 0; c < g.C; c++ {
		colMismatch += abs(sol.colCounts[c] - g.colTarget[c])
	}

	return adjViol + pairViol + rowMismatch + colMismatch
}

// globalPairingViolations computes the pairing penalty using a bipartite matching between placed tents and trees.
func (g *Grid) globalPairingViolations(sol *Solution) int {
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
	return (len(tents) - matchingSize) + (len(trees) - matchingSize)
}

// -----------------------------
// Flexible (Future) Heuristic
// -----------------------------

// futureFlexibilityHeuristic scores a candidate cell (r,c) by giving a bonus for nearby trees
// while subtracting a cost equal to the number of available neighbor cells that would be blocked
// if a tent is placed here. (Cardinal neighbors get a bonus of 2.0 and diagonal ones 1.0.)
// This score is used solely for candidate selection; it is not added to the overall violation count.
func futureFlexibilityHeuristic(r, c int, sol *Solution, g *Grid) float64 {
	bonus := 0.0
	cardinals := [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for _, d := range cardinals {
		nr, nc := r+d[0], c+d[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			bonus += 2.0
		}
	}
	diagonals := [][2]int{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
	for _, d := range diagonals {
		nr, nc := r+d[0], c+d[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			bonus += 1.0
		}
	}

	cost := 0.0
	// Count available (blank and unoccupied) neighbor cells that will be blocked.
	for dr := -1; dr <= 1; dr++ {
		for dc := -1; dc <= 1; dc++ {
			if dr == 0 && dc == 0 {
				continue
			}
			nr, nc := r+dr, c+dc
			if g.inBounds(nr, nc) && g.cells[nr][nc] == '.' && !sol.placements[nr][nc] {
				cost += 1.0
			}
		}
	}
	return bonus - cost
}

// -----------------------------
// Other Heuristic Functions
// -----------------------------

// adjacencyHeuristic focuses on avoiding adjacent tents.
func adjacencyHeuristic(r, c int, sol *Solution, g *Grid) float64 {
	dirs8 := [][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
	for _, d := range dirs8 {
		nr, nc := r+d[0], c+d[1]
		if g.inBounds(nr, nc) && sol.placements[nr][nc] {
			return -1.0
		}
	}
	return 0.0
}

// treeMatchingHeuristic focuses on good tree–tent pairing.
func treeMatchingHeuristic(r, c int, sol *Solution, g *Grid) float64 {
	score := 0.0
	if g.inBounds(r-1, c) && g.cells[r-1][c] == 'T' {
		score++
	}
	if g.inBounds(r+1, c) && g.cells[r+1][c] == 'T' {
		score++
	}
	if g.inBounds(r, c-1) && g.cells[r][c-1] == 'T' {
		score++
	}
	if g.inBounds(r, c+1) && g.cells[r][c+1] == 'T' {
		score++
	}
	return score
}

// rowColumnHeuristic focuses on meeting row and column targets.
func rowColumnHeuristic(r, c int, sol *Solution, g *Grid) float64 {
	rowDef := float64(g.rowTarget[r] - sol.rowCounts[r])
	colDef := float64(g.colTarget[c] - sol.colCounts[c])
	if rowDef < 0 {
		rowDef = 0
	}
	if colDef < 0 {
		colDef = 0
	}
	return rowDef + colDef
}

// -----------------------------
// Generic Greedy Construction
// -----------------------------

// HeuristicFunc is a type for candidate scoring functions.
type HeuristicFunc func(r, c int, sol *Solution, g *Grid) float64

// constructSolution uses the provided heuristic function to guide greedy placement.
// It places tents one by one (without exceeding row/column targets) using the candidate with the highest score.
// Once completed, the overall violation count is calculated using the full evaluation function.
func (g *Grid) constructSolution(h HeuristicFunc) *Solution {
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}
	improvement := true
	for improvement {
		improvement = false
		bestScore := -math.MaxFloat64
		bestR, bestC := -1, -1
		for r := 0; r < g.R; r++ {
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] != '.' {
					continue
				}
				if sol.placements[r][c] {
					continue
				}
				if sol.rowCounts[r] >= g.rowTarget[r] || sol.colCounts[c] >= g.colTarget[c] {
					continue
				}
				score := h(r, c, sol, g)
				if score > bestScore {
					bestScore = score
					bestR = r
					bestC = c
				}
			}
		}
		if bestR != -1 && bestC != -1 {
			sol.placements[bestR][bestC] = true
			sol.rowCounts[bestR]++
			sol.colCounts[bestC]++
			improvement = true
		}
	}
	// IMPORTANT: The overall violation calculation follows the full problem rules.
	sol.violations = g.evaluate(sol)
	return sol
}

// -----------------------------
// Bipartite Matching & Output Direction
// -----------------------------

// computeBipartiteMatchingDetailed pairs each placed tent with an adjacent tree (cardinally).
// It returns a slice matchTent (where matchTent[i] is the index of the tree paired with the i-th tent)
// as well as slices of tent and tree positions.
func (g *Grid) computeBipartiteMatchingDetailed(sol *Solution) (matchTent []int, tents []pos, trees []pos) {
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
	for u := 0; u < len(tents); u++ {
		visited := make([]bool, len(trees))
		_ = dfs(u, visited)
	}
	matchTent = make([]int, len(tents))
	for i := range matchTent {
		matchTent[i] = -1
	}
	for j, i := range matchTree {
		if i != -1 {
			matchTent[i] = j
		}
	}
	return matchTent, tents, trees
}

// computeDirection returns a direction ('U', 'D', 'L', 'R', or 'X') for a tent paired with a tree.
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

// -----------------------------
// Main Function
// -----------------------------

func main() {
	// Expect: first argument = input file, second argument = heuristic type ("adjacency", "tree", "row", "future")
	var inputReader *bufio.Reader
	var inputFileName string
	heuristicType := "adjacency" // default
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
	if len(os.Args) > 2 {
		heuristicType = os.Args[2]
	}

	grid, err := parseInput(inputReader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input: %v\n", err)
		return
	}

	// Select the heuristic based on the argument.
	var h func(r, c int, sol *Solution, g *Grid) float64
	switch strings.ToLower(heuristicType) {
	case "adjacency":
		h = adjacencyHeuristic
	case "tree":
		h = treeMatchingHeuristic
	case "row":
		h = rowColumnHeuristic
	case "future":
		h = futureFlexibilityHeuristic
	default:
		fmt.Fprintf(os.Stderr, "Unknown heuristic type '%s'. Using default (adjacency).\n", heuristicType)
		h = adjacencyHeuristic
	}

	start := time.Now()
	sol := grid.constructSolution(h)
	elapsed := time.Since(start)
	fmt.Printf("Solution constructed in %v using %s heuristic\n", elapsed, heuristicType)
	fmt.Printf("Overall violations: %d\n", sol.violations)

	// Perform bipartite matching for output directions.
	matchTent, tents, trees := grid.computeBipartiteMatchingDetailed(sol)
	outputBuilder := &strings.Builder{}
	tentCount := 0
	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if sol.placements[r][c] {
				tentCount++
			}
		}
	}
	// Output overall violation count and tent count.
	fmt.Fprintf(outputBuilder, "%d\n", sol.violations)
	fmt.Fprintf(outputBuilder, "%d\n", tentCount)
	// Output each tent's (1-indexed) position and assigned direction.
	for i, tpos := range tents {
		tentR, tentC := tpos.r, tpos.c
		treeIdx := matchTent[i]
		dir := 'X'
		if treeIdx != -1 {
			tr, tc := trees[treeIdx].r, trees[treeIdx].c
			dir = computeDirection(tentR, tentC, tr, tc)
		}
		fmt.Fprintf(outputBuilder, "%d %d %c\n", tentR+1, tentC+1, dir)
	}

	// Save output in an "argument" folder with the heuristic type in the filename.
	outputFolder := "argument"
	os.MkdirAll(outputFolder, os.ModePerm)
	baseName := filepath.Base(inputFileName)
	outputFileName := filepath.Join(outputFolder, "output_"+heuristicType+"_"+baseName)
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
