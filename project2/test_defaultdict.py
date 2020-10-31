from utils import *

N = defaultdict(int)
n = defaultdict(int)
N[(1,1,1)] += 1
N[(1,2,1)] += 1
N[(1,1,1)] += 1

for i in range(1, 2):
    for j in range(1,3):
        for k in range(1,2):
            n[(i,j)] += N[(i,j,k)]
            
print n
print N
            
        
            