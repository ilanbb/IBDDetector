import math

def cnk(n, k, mem=None):
    if not mem:
        mem = dict()
    if k == 0:
        return 1
    if n == 0:
        return 0
    if n == k:
        return 1
    if (n,k) not in mem:
        mem[(n,k)] = cnk(n-1, k, mem) + cnk(n-1, k-1, mem)
    return mem[(n,k)]


class BinomialDist:

    def __init__(self, n ,p):
        self.n = n
        self.p = p
        self.binoms = [0] * (n+1)
        for k in range(n/2+1):
            d = cnk(n, k)
            self.binoms[k] = d
            self.binoms[n-k] = d            
        for k in range(n+1):
            self.binoms[k] *= p**k * (1-p)**(n-k)

    def prob_ge(self, t):
       prob = 0
       for i in range(t, self.n+1):
          prob += self.binoms[i]
       return prob
      
Q = 10000.0
K = 50
BPS = Q/K

M = 95000
G = 50*1
SEGMENT_NUM = (M-Q)/G

# number of maximal errors in similar segments
e = 5.0
# number of minimal errors in dissimilar strings (by factor from r)
E = 100.0
# Length of strings
Q = 10000
# One bit hash lower bound on probability of match between similar strings
p1 = (1 - e/Q)
# One bit hash upper bound on probability of match between dissimilar strings
p2 = (1 - E/Q)

# Parameters: T (AND of BPS*T), L (number of masks), C (min count for success)
for L in range(4, 24, 2):
    for T in range(1,K+1):
        X = BinomialDist(L, p1**(BPS*T))
        Y = BinomialDist(L, p2**(BPS*T))
        for C in range(1, L+1):
            tp = X.prob_ge(C)
            fp = Y.prob_ge(C)
            if tp > 0.95 and fp < 0.005:
                print "L:", L, "T:", T, "C:", C, "P1", tp, "P2:", fp
