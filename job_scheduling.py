import pulp

# Option for type of solver?
# Needs function for updating weights
# Should redfine x i think 
class JobScheduling:
    def __init__(self,processing_times,n_jobs,n_machines):
        self.processing_times = processing_times
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.model_lp = None
        self.model_ilp = None
        self.x = None
        self.C = None
    def define_pulp_lp_problem(self):
        model = pulp.LpProblem("Job_Scheduling_Makespan_Minimization", pulp.LpMinimize)

        # Define binary variables: x[i][j] = 1 if job i is assigned to machine j
        self.x = pulp.LpVariable.dicts(
            "Assignment", 
            ((i, j) for i in range(self.n_jobs) for j in range(self.n_machines)), 
            cat=pulp.LpContinuous
        )

        # Makespan variable (continuous)
        self.C = pulp.LpVariable("Makespan", lowBound=0, cat=pulp.LpContinuous)

        # Objective: Minimize the makespan
        model += self.C

        for i in range(n_jobs):
            for j in range(n_machines):
                model += self.x[i,j] >= 0
                model += self.x[i,j] <= 1


        # Constraints
        for i in range(n_jobs):
            # Each job must be assigned to exactly one machine
            model += pulp.lpSum(self.x[i, j] for j in range(self.n_machines)) == 1

        for j in range(n_machines):
            # Total time on machine j must not exceed makespan C
            model += pulp.lpSum(self.x[i, j] * self.processing_times[i, j] for i in range(self.n_jobs)) - self.C<= 0 
        self.model = model

    def define_pulp_problem(self):
        model = pulp.LpProblem("Job_Scheduling_Makespan_Minimization", pulp.LpMinimize)

        # Define binary variables: x[i][j] = 1 if job i is assigned to machine j
        self.x = pulp.LpVariable.dicts(
            "Assignment", 
            ((i, j) for i in range(self.n_jobs) for j in range(self.n_machines)), 
            cat=pulp.LpBinary
        )

        # Makespan variable (continuous)
        self.C = pulp.LpVariable("Makespan", lowBound=0, cat=pulp.LpContinuous)

        # Objective: Minimize the makespan
        model += self.C

        # Constraints
        for i in range(n_jobs):
            # Each job must be assigned to exactly one machine
            model += pulp.lpSum(self.x[i, j] for j in range(self.n_machines)) == 1

        for j in range(n_machines):
            # Total time on machine j must not exceed makespan C
            model += pulp.lpSum(self.x[i, j] * self.processing_times[i, j] for i in range(self.n_jobs)) - self.C <= 0
        self.model = model

    def solve(self):
        self.model.solve()
    # Sample data: 3 jobs, 2 machines





    
n_jobs = 3
n_machines = 2

# Processing times: p[i][j] = time for job i on machine j
processing_times = {
    (0, 0): 2, (0, 1): 4,  # Job 0 on machines 0 and 1
    (1, 0): 3, (1, 1): 1,  # Job 1
    (2, 0): 5, (2, 1): 2   # Job 2
}

prob = JobScheduling(processing_times,n_jobs,n_machines)
prob.define_pulp_lp_problem()
prob.solve()

solver_list = pulp.listSolvers(onlyAvailable=True)
print(solver_list)
print("Duals")
for _,constraint in prob.model.constraints.items():
    print(constraint,constraint.pi)
print("Assignments")
for var in prob.model.variables():
    print(f"{var.name}: {var.value()}")

print(pulp.value(prob.x[0,0]))
print(prob.model.constraints.items())


