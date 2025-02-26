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

// Grid holds the problem data.
type Grid struct {
	R, C      int
	cells     [][]rune // grid of '.' or 'T'
	rowTarget []int
	colTarget []int
}

// TentPlacement represents one tent placement (used for output).
type TentPlacement struct {
	r, c int
	dir  rune // U, D, L, R, or X
}

// Solution holds a candidate solution.
type Solution struct {
	placements [][]bool // same dimensions as grid; true means a tent is placed
	violations int

	// Bookkeeping: current tent counts for rows and columns.
	rowCounts []int
	colCounts []int
}

// -----------------------------
// Parsing
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

// -----------------------------
// Heuristic and Utility Functions
// -----------------------------

// heuristic computes a cell’s heuristic value.
// It rewards cells with adjacent trees and boosts the value if the corresponding row/column are under target.
func (g *Grid) heuristic(r, c int, rowCount, colCount int) float64 {
	score := 0.1
	neighbors := [][2]int{{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}
	for _, n := range neighbors {
		nr, nc := n[0], n[1]
		if g.inBounds(nr, nc) && g.cells[nr][nc] == 'T' {
			score += 1.0
		}
	}
	// Boost if row/column are underfilled.
	rowDeficit := float64(g.rowTarget[r] - rowCount)
	colDeficit := float64(g.colTarget[c] - colCount)
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

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// -----------------------------
// Bipartite Matching (for final violation calculation)
// -----------------------------

// globalPairingViolations computes the pairing penalty using a global matching optimizer.
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
	penalty := (len(tents) - matchingSize) + (len(trees) - matchingSize)
	return penalty
}

// computeBipartiteMatchingDetailed returns the exact matching between tents and trees.
// It returns matchTent where matchTent[i] is the index in trees that is matched with tent i (or -1 if unmatched),
// along with slices of positions for tents and trees.
func (g *Grid) computeBipartiteMatchingDetailed(sol *Solution) (matchTent []int, tents []pos, trees []pos) {
	// Gather tents.
	for r := 0; r < g.R; r++ {
		for c := 0; c < g.C; c++ {
			if sol.placements[r][c] {
				tents = append(tents, pos{r, c})
			}
		}
	}
	// Gather trees.
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

// computeDirection returns a direction character ('U', 'D', 'L', 'R', or 'X')
// for a tent at (tentR,tentC) paired with a tree at (treeR,treeC).
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
// Evaluation Function
// -----------------------------

// evaluate computes the overall violation score of a solution.
// It includes adjacent tent violations, pairing mismatches, and row/column target deviations.
func (g *Grid) evaluate(sol *Solution) int {
	violations := 0
	// 1. Adjacent tent violations (8-neighbor)
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
						break // count at most one violation per tent
					}
				}
			}
		}
	}
	// 2. Pairing violations (using bipartite matching)
	violations += g.globalPairingViolations(sol)
	// 3. Row and column target mismatches.
	for r := 0; r < g.R; r++ {
		violations += int(math.Abs(float64(sol.rowCounts[r] - g.rowTarget[r])))
	}
	for c := 0; c < g.C; c++ {
		violations += int(math.Abs(float64(sol.colCounts[c] - g.colTarget[c])))
	}
	return violations
}

// -----------------------------
// Placement Construction using Most Constrained Heuristic
// -----------------------------

// canPlace returns true if placing a tent at (r, c) does not violate the non-adjacency rule.
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

// constructMostConstrainedSolution builds a candidate solution by first filling in
// forced moves (rows or columns that have exactly as many available cells as needed)
// and then by always choosing the most constrained row or column (fewest candidates)
// and selecting the best candidate cell in that row/column according to the heuristic.
func (g *Grid) constructMostConstrainedSolution() *Solution {
	sol := &Solution{
		placements: make([][]bool, g.R),
		rowCounts:  make([]int, g.R),
		colCounts:  make([]int, g.C),
	}
	for i := 0; i < g.R; i++ {
		sol.placements[i] = make([]bool, g.C)
	}

	changed := true
	// Continue until no more placements can be made.
	for changed {
		changed = false

		// --- Forced moves in rows ---
		for r := 0; r < g.R; r++ {
			remaining := g.rowTarget[r] - sol.rowCounts[r]
			if remaining <= 0 {
				continue
			}
			var candidateCols []int
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					candidateCols = append(candidateCols, c)
				}
			}
			// If the number of candidates exactly equals the number of tents we need, place them.
			if len(candidateCols) == remaining && remaining > 0 {
				for _, c := range candidateCols {
					sol.placements[r][c] = true
					sol.rowCounts[r]++
					sol.colCounts[c]++
				}
				changed = true
			}
		}

		// --- Forced moves in columns ---
		for c := 0; c < g.C; c++ {
			remaining := g.colTarget[c] - sol.colCounts[c]
			if remaining <= 0 {
				continue
			}
			var candidateRows []int
			for r := 0; r < g.R; r++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					candidateRows = append(candidateRows, r)
				}
			}
			if len(candidateRows) == remaining && remaining > 0 {
				for _, r := range candidateRows {
					sol.placements[r][c] = true
					sol.rowCounts[r]++
					sol.colCounts[c]++
				}
				changed = true
			}
		}

		// If any forced moves were made, restart the loop.
		if changed {
			continue
		}

		// --- Choose the most constrained row or column ---
		bestCandidateCount := math.MaxInt32
		bestIsRow := true // true means best index is a row, false means a column
		bestIndex := -1

		// Examine rows.
		for r := 0; r < g.R; r++ {
			remaining := g.rowTarget[r] - sol.rowCounts[r]
			if remaining <= 0 {
				continue
			}
			candidateCount := 0
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					candidateCount++
				}
			}
			if candidateCount > 0 && candidateCount < bestCandidateCount {
				bestCandidateCount = candidateCount
				bestIsRow = true
				bestIndex = r
			}
		}

		// Examine columns.
		for c := 0; c < g.C; c++ {
			remaining := g.colTarget[c] - sol.colCounts[c]
			if remaining <= 0 {
				continue
			}
			candidateCount := 0
			for r := 0; r < g.R; r++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					candidateCount++
				}
			}
			if candidateCount > 0 && candidateCount < bestCandidateCount {
				bestCandidateCount = candidateCount
				bestIsRow = false
				bestIndex = c
			}
		}

		// If we couldn't find any candidate in any row or column, break out.
		if bestIndex == -1 {
			break
		}

		// --- Place a tent in the most constrained row or column ---
		if bestIsRow {
			r := bestIndex
			bestScore := -1.0
			bestC := -1
			for c := 0; c < g.C; c++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					score := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c])
					if score > bestScore {
						bestScore = score
						bestC = c
					}
				}
			}
			if bestC != -1 {
				sol.placements[r][bestC] = true
				sol.rowCounts[r]++
				sol.colCounts[bestC]++
				changed = true
			}
		} else {
			c := bestIndex
			bestScore := -1.0
			bestR := -1
			for r := 0; r < g.R; r++ {
				if g.cells[r][c] == '.' && !sol.placements[r][c] && sol.canPlace(g, r, c) {
					score := g.heuristic(r, c, sol.rowCounts[r], sol.colCounts[c])
					if score > bestScore {
						bestScore = score
						bestR = r
					}
				}
			}
			if bestR != -1 {
				sol.placements[bestR][c] = true
				sol.rowCounts[bestR]++
				sol.colCounts[c]++
				changed = true
			}
		}
	}

	// Final evaluation of the solution.
	sol.violations = g.evaluate(sol)
	return sol
}

// -----------------------------
// Main
// -----------------------------

func main() {
	// Read input.
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
	// Use the most-constrained heuristic to build a solution.
	sol := grid.constructMostConstrainedSolution()
	elapsed := time.Since(start)
	fmt.Printf("Most-constrained solution constructed in %v\n", elapsed)
	fmt.Printf("Total violations: %d\n", sol.violations)

	// Perform detailed bipartite matching so that each tent is paired with a unique tree.
	matchTent, tents, trees := grid.computeBipartiteMatchingDetailed(sol)

	// Prepare output.
	outputBuilder := &strings.Builder{}
	// Count how many tents.
	tentCount := 0
	for r := 0; r < grid.R; r++ {
		for c := 0; c < grid.C; c++ {
			if sol.placements[r][c] {
				tentCount++
			}
		}
	}
	fmt.Fprintf(outputBuilder, "%d\n", sol.violations)
	fmt.Fprintf(outputBuilder, "%d\n", tentCount)

	// Output the tents with final directions from the matching.
	// The tents slice is in the same order as used in the matching.
	for i, tpos := range tents {
		tentR, tentC := tpos.r, tpos.c
		treeIdx := matchTent[i]
		dir := 'X'
		if treeIdx != -1 {
			tr, tc := trees[treeIdx].r, trees[treeIdx].c
			dir = computeDirection(tentR, tentC, tr, tc)
		}
		// Output 1-indexed coordinates.
		fmt.Fprintf(outputBuilder, "%d %d %c\n", tentR+1, tentC+1, dir)
	}

	baseName := filepath.Base(inputFileName)
	outputFileName := filepath.Join("most_constrained", "output_"+baseName)
	os.MkdirAll("most_constrained", os.ModePerm)
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
