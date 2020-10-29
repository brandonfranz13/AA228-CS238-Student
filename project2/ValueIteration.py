from utils import *

def lookahead(discount, S, A, T, R, s, a):
    return R[s,a] + discount * np.sum(np.asarray([T[s,a,sp] * U[sp] for sp in S]))
    
def backup(discount, S, A, T, R, U, s):
    return np.max(np.asarray([lookahead(discount, S, A, T, R, s, a) for a in A]))