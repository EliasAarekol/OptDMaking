

from pyscipopt import Model, quicksum, multidict , Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING
# scip = Model()
# x = scip.addVar(vtype='I', lb=0, ub=None, name='x')
# y = scip.addVar(vtype='I', lb=0, ub=None, name='y')
# cons_1 = scip.addCons(x + y <= 5, name="cons_1")
# cons_1 = scip.addCons(x + y == 5, name="cons_3")
# scip.setObjective(2 * x + 3 * y, sense="minimize")
# scip.optimize()
# print(scip.getCurrentNode())


class SubproblemEventHandler(Eventhdlr):
    def __init__(self, model):
        Eventhdlr.__init__(model)

    def eventinit(self):
        # Catch NODESOLVED events (after LP is solved at a node)
        self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, event):
        print("event!")
        print("\n" + "="*50)
        print("Subproblem Information")
        print("="*50)
        
        # Get current node information
        current_node = model.getCurrentNode()
        node_number = current_node.getNumber()
        # depth = current_node.getDepth()
        # lower_bound = model.getLowerbound()
        
        print(f"Node Number: {node_number}")
        # print(f"Depth: {depth}")
        # print(f"Lower Bound: {lower_bound:.2f}")
        
        # Get variable bounds at this node
        print("\nVariable Bounds:")
        vars = self.model.getVars()
        for var in vars:
            name = var.name
            lb = var.getLbLocal()  # Lower bound at current node
            ub = var.getUbLocal()  # Upper bound at current node
            print(f"{name}: [{lb:.2f}, {ub:.2f}]")
        print(model.getConss())
            
class BestSolCounter(Eventhdlr):
    def __init__(self, model):
        Eventhdlr.__init__(model)
        self.nodes = []

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEEVENT, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODEEVENT, self)

    def eventexec(self, event):
        if model.getCurrentNode() not in self.nodes:
            self.nodes.append(model.getCurrentNode())
        print("New node added")

# I, d = multidict({1:80, 2:270, 3:250, 4:160, 5:180})
# J, M, f = multidict({1:[500,1000], 2:[500,1000], 3:[500,1000]})
# c = {(1,1):4,  (1,2):6,  (1,3):9,
#     (2,1):5,  (2,2):4,  (2,3):7,
#     (3,1):6,  (3,2):3,  (3,3):4,
#     (4,1):8,  (4,2):5,  (4,3):3,
#     (5,1):10, (5,2):8,  (5,3):4,
#     }
# model = Model("flp")
# x,y = {},{}
# for j in J:
#     y[j] = model.addVar(vtype="B", name="y(%s)"%j)
#     for i in I:
#         x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))
# for i in I:
#     model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)
# for j in M:
#     model.addCons(quicksum(x[i,j] for i in I) <= M[j]*y[j], "Capacity(%s)"%i)
# for (i,j) in x:
#     model.addCons(x[i,j] <= d[i]*y[j], "Strong(%s,%s)"%(i,j))
# model.setObjective(
#     quicksum(f[j]*y[j] for j in J) +
#     quicksum(c[i,j]*x[i,j] for i in I for j in J),
#     "minimize")
# model.data = x,y


model = Model()

x = model.addVar(vtype='I', lb=0, ub=None, name='x')
y = model.addVar(vtype='I', lb=0, ub=None, name='y')
y2 = model.addVar(vtype='I', lb=0, ub=None, name='y2')
y3 = model.addVar(vtype='I', lb=0, ub=None, name='y3')
y4 = model.addVar(vtype='I', lb=0, ub=None, name='y4')

cons_1 = model.addCons(x + y +y2 + y3+y4 <= 10, name="cons_1")
cons_1 = model.addCons(4*x + 7*y  + y2 + y3 <= 28, name="cons_3")
cons_1 = model.addCons(4*x + 7*y2  + y4 + y3 <= 30, name="cons_4")
cons_1 = model.addCons(7*x  + y2 + y <= 28, name="cons_5")
best_sol_counter = SubproblemEventHandler(model)
vars = model.getVars()
for var in vars:
    name = var.name
    lb = var.getLbLocal()  # Lower bound at current node
    ub = var.getUbLocal()  # Upper bound at current node
    print(f"{name}: [{lb:.2f}, {ub:.2f}]")

model.includeEventhdlr(best_sol_counter, "best_sol_event_handler", "Event handler that counts the number of best solutions found")
model.setObjective(5* x + 6 * y + y2 + y3 + 2*y4, sense="maximize")
model.setPresolve(SCIP_PARAMSETTING.OFF)
model.setHeuristics(SCIP_PARAMSETTING.OFF)
model.setSeparating(SCIP_PARAMSETTING.OFF)

model.optimize()
print(model.getConss())


# print(best_sol_counter.nodes[0].getNumber())
# model.printStatistics()
