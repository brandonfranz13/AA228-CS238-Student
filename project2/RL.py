from utils import *

# MDP = namedtuple("MDP", "discount States Actions Transition Reward")
 
class SmallRL(discount=0.95, max_iterations=100):
    '''
    Description of Data Set
    10 x 10 grid world (100 states) with 4 actions. Actions are 1: left, 2: right, 3: up, 4: down. The discount factor is 0.95.
    '''

    def __init__(self):
        _, sarsp = importCSV("data/small.csv")
        self.s = sarsp[:,0]
        self.a = sarsp[:,1]
        self.r = sarsp[:,2]
        self.sp = sarsp[:,3]
        self.n_states = 100
        self.n_actions = 4
        self.k = max_iterations
        
        self.N = np.zeros(n_states*n_actions*len(self.s)) # S x A x transitions
        self.rho = np.zeros(n_states*n_actions) # S x A
        
    def single2multi_index(single, dims):
        return np.unravel_index(single, dims)
    
    def multi2single_index(multi, dims):
        return np.ravel_multi_index(multi, dims)
    
    def learnMaxLike(self):
        # Count the occurrence of (s,a,sp) transitions
        for i in range(len(self.s)):
            index = self.multi2single_index(
        # Accumulate the reward for transitions from (s, a)
        return True
    
    def valueIteration(self):
        U = np.zeros(len(self.s))
        for i in range(k):
            U = [backup(
        return True
    
    def restart(self):
        return True
    
    def solve(self):
        writePolicy("small.policy", self.a)
        print "Policy Written to File"
