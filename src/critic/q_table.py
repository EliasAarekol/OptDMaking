from src.critic.critic_template import Critic
import src.utils.q_table as q

class Q_table(Critic):
    def __init__(self,table,rs,lr,df,eps):
        super().__init__()
        self.table = table
        self.rs = rs
        self.lr = lr
        self.df = df 
        self.eps = eps


    def train(self):
        self.table = q.train_q_table(self.table,self.rs,self.lr,self.df,self.eps)
        return self.table
    
    def evaluate(self, actions, states):
        return self.table[actions,states]