# OptDMaking

### TODO

 - <s> Look at solvers structure, right now you need a new object for each node which sucks</S>
 - <s> Don't know if rewards from gym models should be positive or negative </s>
 - Implement gym mode properly so that can use gymnasium.make
 - <s> implement stochastic policy - need soluition pool </s>
 - <s> solution pool for solvers </s>
 - <s> solution pool for brute </s>
 - <s> implement advantage table </s>
 - <s> implement training</s>
 - <s> implement experience buffer </s>
 - rethink and restructure solvers
 - <s> proper seeding in gymnasium reset </s>
 - check intensity calculations (sanity check)
 - What do we actually need to store from solvers
    - solution
    - marginals/dual solution
    - obj_func
 - <s> Rewrite solvers to not be OOP </s>
 - <s> Maybe wrap solvers into policy taker or something </s>
 - <s> How to handle end of run, think right now its working</s>
 - <s>  L1 relaxation </s>
 - <s> terminate when no more possible </s>
 - Handle action encoding better, right now returned by gym model
 - <s> Handle no feasible solutions to problem solved by lp </s> Shouldnt ever happen
 - For some reason gives 0 reward sometimes, i think gym model doesnt work properly
 - <s> Solve bruteforce with multiprocessing? </s>
 - <s> Redefine step update, perhaps as a self defined function </s>
 - <s> Clean up gradient code </s>
 - Check convergence and size of Q table
 - <s> Look for PPO or similar for training inspiration </s>
 - <s> Rewrite the brute force solver to be asynchronous - works for large problems but not for small </S>
 - <s> Something wrong with the brute force solver, sometimes gives non-integer solutions </s>
 - <s> Investigate cvxpylayers </s>
 - <s> Move pool outside </s>
 - Make gradient code general for LP problem
 -<s> Fix file structure </s>
 - penalty factor needs to be lower than any action you want to be picked, can be learned? I think
 - Think the value function in the 
 - <s>Make actor impl flexible so that i can use different critics and solvers </S>
 - Solver configs should be set at init
 - remove q_Table from utils maybe
 - fix doc for all classes
 - fix abstact class inputs
 - fix knapsack a and b update
 - fix brute force strucutre, remove a lot
 - Assume for now we require an upper bound on every variable, s.t the feasible space isnt infinite
 - Parralleize branch and bound
   Is currently slower than brute force approach
   Actually quite similar to parallell brute force approach
 - Fix config for brute force
 - Naive and nn have similar performance if all parameters have a bound
 - Fathoming sometimes just stops
 - small lr tests seem to experience that fathomed branches have the same conditions and conds on every variable
 - Had wrong sign on reward for breaching constraint (think its fixed now) does much worse
 - Increasing pf in gym model only seems to give lower reward, might point to that its not learning to improve
  - Made an error in the plotting so that it was looking like it was improving :( 
 - Finds optimal solution then gets stuck just doing a rollout of all 0s and then a single move, which is prob why the episodic rewards come so rarely
 - Explicit solution can be wrong due to a too "long" problem can be infeasible
 - Smooth episodic reward actually shows the transition, smoothing in wandb doesnt show this
 - What sebastien said might relate to that the naive approach isnt converging because its not the correct gradient for that action
 