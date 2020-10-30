from utils import *
    
def lookahead(S, N, n, rho, discount, U, s, a): # for one state-action pair
    if n == 0:
        return 0.
    r = rho[(s,a)] / n
    T[ = N[s,a,:] / n
    return r + discount * np.sum([T[s,a,sp]*U[sp] for sp in S])
    
def convertToMDP(S, A, N, rho):
    for s in S:
        for a in A:
            n = np.sum(N[s,a,:])
            if n == 0:
                T[s,a,:] = 0.
                R[s,a] = 0.
            else:
                T[s,a,:] = N[s,a,:] / n
                R[s,a] = rho[s,a] / n
    return T, R


def backup(S, A, N, rho, discount, U, s): #for every state-action pair at one state
    return np.max([lookahead(process, U, s, a) for a in process.A