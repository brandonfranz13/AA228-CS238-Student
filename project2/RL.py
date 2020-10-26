import numpy as np

def writePolicy(filename, actions):
    np.savetxt(filename, actions, delimiter=",")
    
def importCSV(filename):
    contents = pd.read_csv(filename)
    return contents.columns, contents.to_numpy()
    
    
class SmallRL()
    '''
    Description of Data Set
    10 x 10 grid world (100 states) with 4 actions. Actions are 1: left, 2: right, 3: up, 4: down. The discount factor is 0.95.
    '''

    def __init__(self)
        _, sarsp = importCSV("small.csv")
        self.s = sarsp[:,0]
        self.a = sarsp[:,1]
        self.r = sarsp[:,2]
        self.sp = sarsp[:,3]
        
    def solve(self)
        writePolicy("small.policy", self.a)
        print "Policy Written to File"