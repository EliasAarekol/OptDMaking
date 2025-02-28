import numpy as np



def adv(q_table):
    
    """

    Args:
        q_table (_type_): nxm matrix of n actions and m states

    Returns:
        _type_: Table of advantages for the given state
    """
    
    
    V_s = np.max(q_table,axis=0)
    # print(V_s)
    adv = q_table - V_s
    return adv

def q_update(q_table, rs , lr,df,actions,states,nxt_states):
    """_summary_

    Args:
        q_table (_type_): _description_
        rs (_type_): _description_
        lr (_type_): _description_
        df (_type_): _description_
        actions (_type_): _description_
        states (_type_): _description_
        nxt_states (_type_): _description_

    Returns:
        _type_: _description_
    """
    for r,action,state,nxt_state in zip(rs,actions,states,nxt_states):
        # print("reward",r)
        # print("q_update",q_table[action,state]  + lr*(r+df*np.max(q_table,axis=0)[nxt_state] - q_table[action,state]  ))
        # print("reward",q_table)
        # if action == 0 and state == 30:
        #     print("Wth")
        #     print(nxt_state)
        #     print(r)
        q_table[action,state] = q_table[action,state]  + lr*(r+df*np.max(q_table,axis=0)[nxt_state] - q_table[action,state]  )
    return q_table
    
    
    
# q = np.random.uniform(0,1,size=(4,3))
# print(q)
    
    
# adv = state_advs(q,None)
# print(adv)