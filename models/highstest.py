import highspy
import numpy as np
from scipy.optimize import linprog
import time


# model = highspy.Highs()

# values = np.array([1, 2, 2, 5, 1])
# weights = np.array([2, 3, 1, 4, 1])
# capacity = 10
# n = len(values)

# # Initialize the HiGHS model

# # Define objective coefficients (negate for maximization)
# obj_coeffs = (-values).tolist()

# # Define variable bounds (binary 0-1)
# lower_bounds = [0] * n
# upper_bounds = [1] * n

# # Define constraint matrix
# num_nz = n  # Number of nonzero elements in constraint matrix
# constraint_matrix_starts = list(range(n + 1))  # Column-wise storage format
# constraint_indices = [0] * n  # All constraints reference the first row
# constraint_values = weights.tolist()  # Weights of items

# # Add variables (columns)
# model.addCols(
#     n,              # Number of variables
#     obj_coeffs,     # Objective coefficients
#     lower_bounds,   # Lower bounds (0)
#     upper_bounds,   # Upper bounds (1)
#     num_nz,         # Number of nonzeros in constraint matrix
#     constraint_matrix_starts,
#     constraint_indices,
#     constraint_values
# )

# # Add constraint (rows)
# constraint_sense = [capacity]  # Upper bound of the weight constraint
# model.addRows(
#     1,  # Number of constraints
#     [-float('inf')],  # Lower bound (-inf, since it's an inequality)
#     constraint_sense,  # Upper bound (capacity)
#     num_nz,  # Number of nonzeros
#     constraint_indices,
#     list(range(n)),
#     constraint_values
# )

# # Solve the problem
# model.run()
# # print(model.HighsLp())
# # Extract and print the solution
# solution = model.getSolution()
# print("Optimal solution:", solution)
# n = len(values)

# # print(sol)

# values = np.array([1, 2, 2, 5, 1])
# weights = np.array([2, 3, 1, 4, 1])
# capacity = 10
# n = len(values)


# # Define the problem in highspy
# h = highspy.Highs()

# # Objective function
# h.changeColCost(0, -1)  # Coefficient for x1
# h.changeColCost(1, 4)   # Coefficient for x2

# # Inequality constraints
# h.addRow(-highspy.kHighsInf, 6, 2, [0, 1], [-3, 1])  # First constraint: -3*x1 + x2 <= 6
# h.addRow(-highspy.kHighsInf, 4, 2, [0, 1], [1, 2])   # Second constraint: x1 + 2*x2 <= 4

# # Variable bounds
# h.changeColBounds(0, 0, highspy.kHighsInf)  # 0 <= x1 <= inf
# h.changeColBounds(1, 0, highspy.kHighsInf)  # 0 <= x2 <= inf

# # Solve the problem
# h.run()
# solution = h.getSolution()


def translate_to_highspy(c, A_ub, b_ub, bounds=None):
    """
    Translate a linear programming problem defined by A_ub and b_ub into highspy format.

    Parameters:
        c (np.array): Coefficient vector of the objective function.
        A_ub (np.array): Inequality constraint matrix.
        b_ub (np.array): Inequality constraint vector.
        bounds (list of tuples): Bounds for each variable, e.g., [(0, None), (0, None)].

    Returns:
        h (highspy.Highs): A Highs object with the problem defined.
    """
    # Initialize the Highs object
    h = highspy.Highs()

    # Add variables (columns) to the model with their bounds
    num_vars = len(c)
    if bounds is None:
        # Default bounds: 0 <= x <= inf
        bounds = [(0, None)] * num_vars

    for i in range(num_vars):
        lb, ub = bounds[i]
        # Convert None to highspy.kHighsInf for unbounded variables
        lb = -highspy.kHighsInf if lb is None else lb
        ub = highspy.kHighsInf if ub is None else ub
        h.addVar(lb, ub)  # Add variable with bounds

    # Add the objective function
    for i in range(num_vars):
        h.changeColCost(i, c[i])

    # Add the inequality constraints
    num_constraints, num_vars = A_ub.shape
    for i in range(num_constraints):
        # Extract the coefficients for the current constraint
        coefficients = A_ub[i, :]
        # Add the constraint: -inf <= A_ub[i] * x <= b_ub[i]
        h.addRow(-highspy.kHighsInf, b_ub[i], num_vars, list(range(num_vars)), coefficients.tolist())

    return h
# Example usage
if __name__ == "__main__":


    values = np.array([1, 2, 2, 5, 1])
    weights = np.array([2, 3, 1, 4, 1])
    capacity = 10
    n = len(values)
    # Define the problem in scipy format
    c = -values   # Maximize -> minimize negative value
    A = np.array([weights])  # Single inequality constraint for total weight
    b = np.array([capacity]) # Knapsack capacity
    bounds = [(0, 1) for _ in range(n)] # Variable bounds

    # Translate to highspy
    h = translate_to_highspy(c, A,b, bounds)

    # Solve the problem
    start = time.time()
    h.run()
    print(time.time()-start)
    solution = h.getSolution()
    print("Optimal solution:", solution.col_value)
    print("Dual", solution.row_dual)
    print(solution.dual_valid)
    start = time.time()
    
    sol = linprog( 
        -values,
        np.atleast_2d(weights),
        capacity,
        None,
        None,
        bounds = [(0, 1) for _ in range(n)],
        )
    print(time.time()-start)
    
    print(sol)
