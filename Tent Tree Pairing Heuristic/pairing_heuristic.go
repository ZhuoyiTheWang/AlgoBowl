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

// TentPlacement holds one tent’s placement (used for final output).
type TentPlacement struct {
	r, c int
	dir  rune // U, D, L, R, or X
}

// Solution represents a candidate solution.
type Solution struct {
	placements [][]bool // true means a tent is placed at that cell
	violations int      // computed pairing violation (from unmatched tents/trees)

	// Bookkeeping for row/column counts.
	rowCounts []int
	colCounts []int
}

// -----------------------------
// Parsing Functions
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

// inBounds returns whether the (r, c) coordinate lies within the grid.
func (g *Grid) inBounds(r, c int) bool {
	return r >= 0 && r < g.R && c >= 0 && c < g.C
}

// -----------------------------
// Tree-Tent Matching Heuristic
// -----------------------------

// treeMatchingHeuristic returns a score for placing a tent at (r, c)
// based solely on the number of adjacent trees (cardinal directions only).
// Each adjacent tree (up, down, left, right) adds 1 point.
func (g *Grid) treeMatchingHeuristic(r, c int, sol *Solution) float64 {
	score := 0.0
	// Check up.
	if g.inBounds(r-1, c) && g.cells[r-1][c] == 'T' {
		score += 1.0
	}
	// Check down.
	if g.inBounds(r+1, c) && g.cells[r+1][c] == 'T' {
		score += 1.0
	}
	// Check left.
	if g.inBounds(r, c-1) && g.cells[r][c-1] == 'T' {
		score += 1.0
	}
	// Check right.
	if g.inBounds(r, c+1) && g.cells[r][c+1] == 'T' {
		score += 1.0
	}
	return score
}

// -----------------------------
// Evaluation (Pairing Only)
// -----------------------------

// globalPairingViolations computes the penalty based on unmatched tents and trees
// using a bipartite matching between placed tents and trees (only cardinal adjacencies).
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
	// Build adjacency: a tent is adjacent to a tree if they touch vertically or horizontally.
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
	// Penalty is the number of unmatched tents plus unmatched trees.
	penalty := (len(tents) - matchingSize) + (len(trees) - matchingSize)
	return penalty
}

// -----------------------------
// Greedy Construction Using Tree-Tent Matching Heuristic
// -----------------------------

// In this construction, we repeatedly scan all blank cells (subject to not exceeding row/column targets)
// and choose the candidate with the highest treeMatchingHeuristic score.
// We only place a tent if the candidate yields a positive score (i.e. it is adjacent to at least one tree).
func (g *Grid) constructTreeMatchingSolution() *Solution {
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
		bestScore := -1.0 // we require a positive score to favor pairing
		bestR, bestC := -1, -1
		for r := 0; r < g.R; r++ {
			for c := 0; c < g.C; c++ {
				// Only consider blank cells.
				if g.cells[r][c] != '.' {
					continue
				}
				// Skip if a tent is already placed.
				if sol.placements[r][c] {
					continue
				}
				// Do not exceed row/column targets.
				if sol.rowCounts[r] >= g.rowTarget[r] || sol.colCounts[c] >= g.colTarget[c] {
					continue
				}
				score := g.treeMatchingHeuristic(r, c, sol)
				if score > bestScore {
					bestScore = score
					bestR = r
					bestC = c
				}
			}
		}
		// Only place a tent if we found a candidate with a positive score.
		if bestR != -1 && bestC != -1 && bestScore > 0 {
			sol.placements[bestR][bestC] = true
			sol.rowCounts[bestR]++
			sol.colCounts[bestC]++
			improvement = true
		}
	}
	// Compute the pairing violation (unmatched tents/trees) as our evaluation.
	sol.violations = g.globalPairingViolations(sol)
	return sol
}

// -----------------------------
// Bipartite Matching & Direction Assignment (for Output)
// -----------------------------

// computeBipartiteMatchingDetailed returns the matching between placed tents and trees.
// It returns matchTent where matchTent[i] is the index in trees matched with tent i (or -1 if unmatched),
// along with the lists of tent and tree positions.
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

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// -----------------------------
// Main Function
// -----------------------------

func main() {
	// Read input from file (if provided) or from standard input.
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

	grid, err := parseInput(inputReader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input: %v\n", err)
		return
	}

	start := time.Now()
	// Construct a solution using the tree-tent matching heuristic.
	sol := grid.constructTreeMatchingSolution()
	elapsed := time.Since(start)
	fmt.Printf("Tree-matching solution constructed in %v\n", elapsed)
	fmt.Printf("Pairing violation (unmatched tents/trees): %d\n", sol.violations)

	// Use bipartite matching to assign each tent an adjacent tree (for output directions).
	matchTent, tents, trees := grid.computeBipartiteMatchingDetailed(sol)

	// Prepare output.
	outputBuilder := &strings.Builder{}
	// Count the number of placed tents.
	tentCount := 0
	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if sol.placements[r][c] {
				tentCount++
			}
		}
	}
	// First two lines: pairing violation and tent count.
	fmt.Fprintf(outputBuilder, "%d\n", sol.violations)
	fmt.Fprintf(outputBuilder, "%d\n", tentCount)

	// For each tent, output its (1-indexed) position and the direction determined by matching.
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
