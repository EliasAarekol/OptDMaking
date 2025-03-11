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
# # solution = h.getSolution()

# import highspy
# import numpy as np

def translate_to_highspy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Translate a linear programming problem defined in scipy format to highspy.

    Parameters:
        c (np.array): Coefficient vector of the objective function.
        A_ub (np.array): Inequality constraint matrix (optional).
        b_ub (np.array): Inequality constraint vector (optional).
        A_eq (np.array): Equality constraint matrix (optional).
        b_eq (np.array): Equality constraint vector (optional).
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

    # Add equality constraints (A_eq x = b_eq)
    if A_eq is not None and b_eq is not None:
        num_eq_constraints, num_vars_eq = A_eq.shape
        assert num_vars_eq == num_vars, "A_eq must have the same number of columns as variables"
        for i in range(num_eq_constraints):
            coefficients = A_eq[i, :]
            # Equality constraint: b_eq[i] <= A_eq[i] * x <= b_eq[i]
            h.addRow(b_eq[i], b_eq[i], num_vars, list(range(num_vars)), coefficients.tolist())

    # Add inequality constraints (A_ub x <= b_ub)
    if A_ub is not None and b_ub is not None:
        num_ub_constraints, num_vars_ub = A_ub.shape
        assert num_vars_ub == num_vars, "A_ub must have the same number of columns as variables"
        for i in range(num_ub_constraints):
            coefficients = A_ub[i, :]
            # Inequality constraint: -inf <= A_ub[i] * x <= b_ub[i]
            h.addRow(-highspy.kHighsInf, b_ub[i], num_vars, list(range(num_vars)), coefficients.tolist())

    return h

# Example usage
if __name__ == "__main__":
    # Define the problem in scipy format
    c = np.array([-1, 4])  # Objective function: minimize -x1 + 4x2

    # Inequality constraints: -3x1 + x2 <= 6 and x1 + 2x2 <= 4
    A_ub = np.array([[-3, 1], [1, 2]])
    b_ub = np.array([6, 3])

    # Equality constraint: x1 + x2 = 2 (optional)
    A_eq = np.array([[1, 1]])
    b_eq = np.array([2])

    # Variable bounds: x1 >= 0, x2 >= 0
    bounds = [(0, None), (0, None)]

    # Translate to highspy
    start = time.time()
    h = translate_to_highspy(-c, A_ub, b_ub, A_eq, b_eq, bounds)

    # Solve the problem
    h.run()
    print(time.time()-start)

    # Retrieve solutions
    solution = h.getSolution()
    primal_solution = solution.col_value  # Primal solution (x)
    dual_solution = solution.row_dual    # Dual solution (shadow prices)
    reduced_costs = solution.col_dual     # Reduced costs (related to variable bounds)

    # Print results
    print("Optimal primal solution (x):", primal_solution)
    print("Dual solution (shadow prices):", dual_solution)
    print("Reduced costs:", reduced_costs)
    print("Objective value:", primal_solution @ c)
    
    start = time.time()
    
    sol = linprog( 
        -c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        bounds,
        )
    # print(time.time()-start)
    print(sol)
# def translate_to_highspy(c, A_ub, b_ub, bounds=None):
#     """
#     Translate a linear programming problem defined by A_ub and b_ub into highspy format.

#     Parameters:
#         c (np.array): Coefficient vector of the objective function.
#         A_ub (np.array): Inequality constraint matrix.
#         b_ub (np.array): Inequality constraint vector.
#         bounds (list of tuples): Bounds for each variable, e.g., [(0, None), (0, None)].

#     Returns:
#         h (highspy.Highs): A Highs object with the problem defined.
#     """
#     # Initialize the Highs object
#     h = highspy.Highs()

#     # Add variables (columns) to the model with their bounds
#     num_vars = len(c)
#     if bounds is None:
#         # Default bounds: 0 <= x <= inf
#         bounds = [(0, None)] * num_vars

#     for i in range(num_vars):
#         lb, ub = bounds[i]
#         # Convert None to highspy.kHighsInf for unbounded variables
#         lb = -highspy.kHighsInf if lb is None else lb
#         ub = highspy.kHighsInf if ub is None else ub
#         h.addVar(lb, ub)  # Add variable with bounds

#     # Add the objective function
#     for i in range(num_vars):
#         h.changeColCost(i, c[i])

#     # Add the inequality constraints
#     num_constraints, num_vars = A_ub.shape
#     for i in range(num_constraints):
#         # Extract the coefficients for the current constraint
#         coefficients = A_ub[i, :]
#         # Add the constraint: -inf <= A_ub[i] * x <= b_ub[i]
#         h.addRow(-highspy.kHighsInf, b_ub[i], num_vars, list(range(num_vars)), coefficients.tolist())

#     return h
# # Example usage
# if __name__ == "__main__":


#     values = np.array([1, 2, 2, 5, 1])
#     weights = np.array([2, 3, 1, 4, 1])
#     capacity = 10
#     n = len(values)
#     # Define the problem in scipy format
#     c = -values   # Maximize -> minimize negative value
#     A = np.array([weights])  # Single inequality constraint for total weight
#     b = np.array([capacity]) # Knapsack capacity
#     bounds = [(0, 1) for _ in range(n)] # Variable bounds

#     # Translate to highspy
#     h = translate_to_highspy(c, A,b, bounds)

#     # Solve the problem
#     start = time.time()
#     h.run()
#     print(time.time()-start)
#     solution = h.getSolution()
#     print("Optimal solution:", solution.col_value)
#     print("Dual", solution.row_dual)
#     print(solution.dual_valid)
    
#     print(sol)
