from utils import *

def lookahead(States, Counts, rho, discount, Utility, current_state, current_action):
    S, U, N = States, Utility, Counts
    s, a = current_state, current_action
    n = np.sum(N[s, a, :])
    if n == 0:
        return 0.
    r = rho[s,a] / n
    T = N[s,a,:] / n
    return r + discount * np.sum([T[s,a,sp]*U[sp] for sp in S])
    
def convertToMDP(States, Actions, Counts, rho):
    S, A, N = States, Actions, Counts
    T, R = np.empty_like(N), np.empty_like(rho)
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


def backup(States, Actions, Counts, rho, discount, Utility, current_state):
    return np.max([lookahead(process, Utility, current_state, action) for action in process.Actions