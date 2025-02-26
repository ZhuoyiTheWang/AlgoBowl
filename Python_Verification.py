import sys
import traceback as tb

input_filepath = sys.argv[1]
output_filepath = sys.argv[2]
dims = []
actual_violations = 0
actual_rows = []
actual_cols = []
tent_locs = {}
input_matrix = []
tree_locs = {}
claimed_violations = int()
claimed_tents = int()
actual_tents = int()

def ReadInputFile(filepath):
    global dims, row_reqs, col_reqs, input_matrix, tree_locs, actual_rows, actual_cols
    try:
        with open(filepath) as input_file:
            dims_str = input_file.readline()
            rows_str = input_file.readline()
            cols_str = input_file.readline()
            dims = [int(val) for val in dims_str.strip().split()]
            row_reqs = [int(val) for val in rows_str.strip().split()]
            col_reqs = [int(val) for val in cols_str.strip().split()]
            for i, line in enumerate(input_file):
                row = list(line.strip())
                input_matrix.append(row)
                for j, char in enumerate(input_matrix[i]):
                    if(char == 'T'):
                        tree_locs[(i + 1, j + 1)] = {
                            "row":int(i + 1),
                            "col":int(j+ 1)
                        }
    except:
        raise Exception("Input file could not be read").with_traceback(sys.exception().__traceback__)
    
    actual_rows = [0] * len(row_reqs)
    actual_cols = [0] * len(col_reqs)
    return

def ReadOutputFile(filepath):
    global tent_locs, actual_tents, claimed_violations, claimed_tents
    actual_tents = 0
    try:
        with open(filepath) as output_file:
            claimed_violations = output_file.readline()
            claimed_tents = output_file.readline()
            for line in output_file:
                row, col, tree_dir = line.strip().split()
                tent_locs[(int(row), int(col))] = {
                    'tree_direction':tree_dir
                }
                actual_tents += 1
    except:
        raise Exception("Output file could not be read").with_traceback(sys.exception().__traceback__)
    return

def ValidateTreeAdjacency():
    global tent_locs, tree_locs, actual_violations, actual_rows, actual_cols, dims
    matched_tents = 0
    matched_trees = {}
    for (row, col), tent in tent_locs.items():
        if((row, col) in tree_locs and tree_locs[(row, col)] is not None):
            raise Exception(f"Tent at ({row}, {col}) was placed on top of a tree.")
        if(row < 1 or row > dims[0] or col < 1 or col > dims[1]):
            raise Exception(f"Tent at ({row}, {col}) was placed outside of the grid.")
        actual_rows[(row - 1)] += 1
        actual_cols[(col - 1)] += 1
        current_row_col = (row, col)
        match tent['tree_direction']:
            case "U":
                if((current_row_col[0] - 1, current_row_col[1]) not in tree_locs or tree_locs[(current_row_col[0] - 1, current_row_col[1])] is None):
                    raise Exception(f"Tent at {str(current_row_col)} does not have a tree where specified ({tent['tree_direction']})")
                if((current_row_col[0] - 1, current_row_col[1]) in matched_trees and matched_trees[(current_row_col[0] - 1, current_row_col[1])] is not None):
                    raise Exception(f"Tree at ({current_row_col[0] - 1}, {current_row_col[1]}) is paired with more than one tent.")
                matched_trees[(current_row_col[0] - 1, current_row_col[1])] = {
                    "row":int((current_row_col[0] - 1)),
                    "col":int(current_row_col[1])
                }
            case "D":
                if((current_row_col[0] + 1, current_row_col[1]) not in tree_locs or tree_locs[(current_row_col[0] + 1, current_row_col[1])] is None):
                    raise Exception(f"Tent at {str(current_row_col)} does not have a tree where specified ({tent['tree_direction']})")
                if((current_row_col[0] + 1, current_row_col[1]) in matched_trees and matched_trees[(current_row_col[0] + 1, current_row_col[1])] is not None):
                    raise Exception(f"Tree at ({current_row_col[0] + 1}, {current_row_col[1]}) is paired with more than one tent.")
                matched_trees[(current_row_col[0] + 1, current_row_col[1])] = {
                    "row":int((current_row_col[0] + 1)),
                    "col":int(current_row_col[1])
                }
            case "L":
                if((current_row_col[0], current_row_col[1] - 1) not in tree_locs or tree_locs[(current_row_col[0], current_row_col[1] - 1)] is None):
                    raise Exception(f"Tent at {str(current_row_col)} does not have a tree where specified ({tent['tree_direction']})")
                if((current_row_col[0], current_row_col[1] - 1) in matched_trees and matched_trees[(current_row_col[0], current_row_col[1] - 1)] is not None):
                    raise Exception(f"Tree at ({current_row_col[0]}, {current_row_col[1] - 1}) is paired with more than one tent.")
                matched_trees[(current_row_col[0], current_row_col[1] - 1)] = {
                    "row":int(current_row_col[0]),
                    "col":int((current_row_col[1] - 1))
                }
            case "R":
                if((current_row_col[0], current_row_col[1] + 1) not in tree_locs or tree_locs[(current_row_col[0], current_row_col[1] + 1)] is None):
                    raise Exception(f"Tent at {str(current_row_col)} does not have a tree where specified ({tent['tree_direction']})")
                if((current_row_col[0], current_row_col[1] + 1) in matched_trees and matched_trees[(current_row_col[0], current_row_col[1] + 1)] is not None):
                    raise Exception(f"Tree at ({current_row_col[0]}, {current_row_col[1] + 1}) is paired with more than one tent.")
                matched_trees[(current_row_col[0], current_row_col[1] + 1)] = {
                    "row":int(current_row_col[0]),
                    "col":int((current_row_col[1] + 1))
                }
            case "X":
                actual_violations += 1
            case _:
                raise Exception(f"Tent at {str(current_row_col)} has an invalid tree direction character.")
    matched_trees = {key: value for key, value in tree_locs.items() if key not in matched_trees}
    if(len(matched_trees) > 0):
        for (row, col), tree in matched_trees.items():
            actual_violations += 1
    return

def ValidateColRowCounts():
    global actual_rows, actual_cols, row_reqs, col_reqs, actual_violations
    row_diff = int()
    col_diff = int()
    for i, row in enumerate(actual_rows):
        row_diff = row - row_reqs[i]
        if(row_diff > 0):
            actual_violations += row_diff
        elif(row_diff < 0):
            actual_violations += abs(row_diff)
    for i, col in enumerate(actual_cols):
        col_diff = col - col_reqs[i]
        if(col_diff > 0):
            actual_violations += col_diff
        elif(col_diff < 0):
            actual_violations += abs(col_diff)
    return

def ValidateTentAdjacency():
    global tent_locs, actual_violations
    for (row, col), tent in tent_locs.items():
        current_row_col = (row, col)
        if(((row+1, col) in tent_locs and tent_locs[(row+1, col)] is not None) or ((row+1, col+1) in tent_locs and tent_locs[(row+1, col+1)] is not None) or ((row, col+1) in tent_locs and tent_locs[(row, col+1)] is not None) or ((row-1, col) in tent_locs and tent_locs[(row-1, col)] is not None) or ((row-1, col-1) in tent_locs and tent_locs[(row-1, col-1)] is not None) or ((row, col-1) in tent_locs and tent_locs[(row, col-1)] is not None) or ((row+1, col-1) in tent_locs and tent_locs[(row+1, col-1)] is not None) or ((row-1, col+1) in tent_locs and tent_locs[(row-1, col+1)] is not None)):
            actual_violations += 1
        else:
            continue
    return


try:
    ReadInputFile(input_filepath)
    ReadOutputFile(output_filepath)
    print(f"\n{output_filepath}")
except Exception as e:
    tb.print_exc()
    sys.exit(1)

try:
    ValidateTreeAdjacency()
    if(int(actual_tents) != int(claimed_tents)):
        raise Exception(f"Invalid tent count: Claimed {claimed_tents} tents, but had {actual_tents} tents.")
    ValidateColRowCounts()
    ValidateTentAdjacency()
    if(int(actual_violations) != int(claimed_violations)):
        raise Exception(f"Invalid violations count: Claimed {claimed_violations} violations, but had {actual_violations} violations.")
except Exception as e:
    tb.print_exc()
    exit(1)
print(f"Claimed {int(claimed_violations)} violations, had {actual_violations} violations.")
print(f"Claimed {int(claimed_tents)} tents, had {actual_tents} tents.")
print("Valid")
exit(0)