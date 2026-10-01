import numpy as np
CORNERS=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
def farthest_point(n, candidates=16384, seed=12345):
    """Greedy farthest-point initial design on the unit square."""
    if n<4: raise ValueError('n must be at least 4')
    pool=np.random.default_rng(seed).random((candidates,2)); selected=CORNERS.copy()
    d=np.min(np.linalg.norm(pool[:,None,:]-selected[None,:,:],axis=2),axis=1)
    while len(selected)<n:
        j=int(np.argmax(d)); p=pool[j].copy(); selected=np.vstack([selected,p]); d=np.minimum(d,np.linalg.norm(pool-p,axis=1)); d[j]=-np.inf
    return selected
