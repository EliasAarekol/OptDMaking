from src.critic.critic_template import Critic
import src.utils.q_table as q

class Q_table(Critic):
    def __init__(self,table,lr,df,eps):
        super().__init__()
        self.table = table
        self.lr = lr
        self.df = df 
        self.eps = eps


    def train(self,rewards,actions,states,nxt_states):
        self.table = q.train_q_table(self.table,rewards,self.lr,self.df,actions,states,nxt_states,self.eps,mode = 1)
        return self.table
    
    def evaluate(self, actions, states):
        return self.table[actions,states]