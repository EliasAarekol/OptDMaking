# OptDMaking

### TODO

 - Look at solvers structure, right now you need a new object for each node which sucks
 - <s> Don't know if rewards from gym models should be positive or negative </s>
 - Implement gym mode properly so that can use gymnasium.make
 - <s> implement stochastic policy - need soluition pool </s>
 - solution pool for solvers
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
 - Rewrite solvers to not be OOP
 - Maybe wrap solvers into policy taker or something
 - <s> How to handle end of run, think right now its working</s>
 - L1 relaxation
 - <s> terminate when no more possible </s>
 - Handle action encoding better, right now returned by gym model
 - Handle no feasible solutions to problem solved by lp
 - For some reason gives 0 reward sometimes, i think gym model doesnt work properly
 - Solve bruteforce with multiprocessing?
 - Redefine step update, perhaps as a self defined function 
 - Clean up gradient code
 - Check convergence and size of Q table
 - Look for PPO or similar for training inspiration
 - Rewrite the brute force solver to be asynchronous - works for large problems but not for small
 - Something wrong with the brute force solver, sometimes gives non-integer solutions
 