#//=============================================================
#//(c) 2011 Distributed under MIT-style license. 
#//(see LICENSE.txt or visit http://opensource.org/licenses/MIT)
#//=============================================================

import numpy as np
from itertools import product
	

def mdl(data,arity,a=1):
    """ Minimum Description Length metric as described by Decampos \
        a=1 constitutes standard mdl (BIC) score
        a=0 constitutes AIC score
    """

    if len(arity) == 0:
        return 0  # Edge has no valid arity info
    ri = arity[0]
    sink = data[:,0]
    N = sink.size
    penalty = (ri-1)*(np.log(N)*.5*a+1-a)

    if data.shape[1]==1:
        abasis = np.array([0])
        delta_complexity = 0.0
    else:
        arity[0] = 1
        abasis = np.concatenate(( [0], np.multiply.accumulate(arity[:-1]) ))
        arity[0] = ri
        qi = np.prod(arity[1:])
        penalty = penalty*qi

    drepr = np.dot( data, abasis )
    un=np.unique( drepr )
    
    #Nijk = np.zeros( ( un.size, ri ), dtype=int )
    #for k,v in enumerate( un ):
    #    Nijk[ k, : ] = np.bincount( sink[ drepr == v ].astype( int ), minlength=ri ) 

    Nijk=np.array([np.bincount(sink[drepr==v].astype(int),minlength=ri) for v in un])


    Nij = np.sum(Nijk,axis=1)
    I = Nij>0
    Nijk = Nijk[I,:]
    pijk = Nijk/Nij[I].reshape(Nij.size,1)
    LL = sum(Nijk[pijk>0]*np.log(pijk[pijk>0]))

    #LL = -H/N
    #H(X|Y) = H(X,Y) - H(Y)
    #MI(X,Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y) 
    #penalty=penalty*(1+np.log(alpha)/np.log(N))/alpha 
    return -LL/N + penalty/N, 0.0
    #return -LL/N, penalty/N


def cpt(data,arity):
    """ Get conditional probability counts, state configurations, 
        encoding basis, relative entropy and complexity values
    """

    ri=arity[0]
    sink=data[:,0]

    if data.shape[1]==1:
        abasis=np.array([0])
        cartesian = np.array([])
        un=np.array([0])
    else:
        arity[0]=1
        abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
        arity[0]=ri
        states = [np.arange(r,dtype=np.int8) for r in arity[1:]]
        cartesian = np.array([i for i in product(*states)])
        un=np.sort(np.dot(cartesian,abasis[1:]))

    drepr=np.dot(data,abasis)
    #un1=np.dot(cartesian,abasis[1:])
    #un=np.unique(drepr)
    w_ij=dict([(val,ind) for ind,val in enumerate(un)])
    v_ik=np.unique(sink)
    Nijk=np.zeros((un.size,ri),dtype=int)
    #for k,v in enumerate(un):
    #   Nijk[k,:]=np.bincount(cnode[drepr==v].astype(int),minlength=ri)
    Nijk=np.array([np.bincount(sink[drepr==v].astype(int),minlength=ri) for v in un])
    Nij=np.sum(Nijk,axis=1)
    N_ij=Nij[Nij!=0]
    N_ijk=Nijk[Nijk!=0]
    H=np.dot(N_ijk,np.log(N_ijk))\
        -np.dot(N_ij[N_ij.nonzero()],np.log(N_ij[N_ij.nonzero()]))
    C=(ri-1)*np.multiply.reduce(arity[1:])*np.log(sink.size)*.5

    #pstates=[data[np.where(drepr==q)[0][0],1:] for q in un]

    return Nijk,Nij,cartesian, abasis, H, C

def bdm(data,arity,new_ancestor_arity=0):
    """ Bayesian Dirichlet metric as described by Cooper&Herskovits """
    N=data.shape[0]
    csize=data.shape[0]+max(arity)
    lgcache=np.arange(0,csize)
    lgcache[0]=1
    lgcache=np.log(lgcache)
    lgcache[0]=0.0
    lgcache=np.add.accumulate(lgcache)
    ri=arity[0]
    cnode=data[:,0]

    if data.shape[1]==1:
        abasis=np.array([0])
    else:
        arity[0]=1
        abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
        arity[0]=ri

    drepr=np.dot(data,abasis)
    un=np.unique(drepr)
    Nijk=np.zeros((un.size,ri),dtype=int)
    for k,v in enumerate(un):
        Nijk[k,:]=np.bincount(cnode[drepr==v].astype(int),minlength=ri)

    N_ij=np.sum(Nijk,axis=1)
    BD = np.sum(lgcache[ri-1]+np.sum(lgcache[Nijk],axis=1)-lgcache[N_ij+ri-1])
    return -BD/N, 0.0


def mu(data,arity,new_ancestor_arity=0):
    """ Minimum Uncertainty (Mutual Information with sampling uncertainty control)"""

    ri=arity[0]
    sink=data[:,0]
    qi=0.0

    if data.shape[1]==1:
        abasis=np.array([0])
    else:
        arity[0]=1
        abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
        arity[0]=ri
        qi = np.prod(arity[1:])

    drepr=np.dot(data,abasis)
    un=np.unique(drepr)
    #Nijk=np.zeros((un.size,ri),dtype=int)
    #for k,v in enumerate(un):
    #    Nijk[k,:]=np.bincount(cnode[drepr==v].astype(int),minlength=ri)

    Nijk=np.array([np.bincount(sink[drepr==v].astype(int),minlength=ri) for v in un])

    Nij = np.sum(Nijk,axis=1)
    I = Nij>0
    Nijk = Nijk[I,:]
    pijk = Nijk/Nij[I].reshape(Nij.size,1)
    LL = sum(Nijk[pijk>0]*np.log(pijk[pijk>0]))

    #other MU ###########
    #term1  = Nij - ri + 1
    #term0 = term1 + 1
    #term0=term0[term1>0]
    #term1=term1[term1>0]
    #mu =  sum( term0*np.log(term0)\
    #        - term1*np.log(term1))\
    #        + qi*(ri-2)*np.log(2)
    #######################

    
    # original MU ###
    XI = 2 + np.pi**2/24
    mu = sum(np.log(Nij+1))+qi*XI
    if qi==0.0: mu=0.0
    ############################
    N=sink.size

    #LL = -H/N
    #H(X|Y) = H(X,Y) - H(Y)
    #MI(X,Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y) 
    return -LL/N + mu/N, 0.0
    #return -LL/N,  mu/N


def stirling(data,arity,new_ancestor_arity=0):
    """ stirling approx. of bdm """
    ri=arity[0]
    sink=data[:,0]

    if data.shape[1]==1:
        abasis=np.array([0])
    else:
        arity[0]=1
        abasis=np.concatenate(([0],np.multiply.accumulate(arity[:-1])))
        arity[0]=ri

    drepr=np.dot(data,abasis)
    un=np.unique(drepr)
    #Nijk=np.zeros((un.size,ri),dtype=int)
    #for k,v in enumerate(un):
    #    Nijk[k,:]=np.bincount(cnode[drepr==v].astype(int),minlength=ri)

    Nijk=np.array([np.bincount(sink[drepr==v].astype(int),minlength=ri) for v in un])

    Nij = np.sum(Nijk,axis=1)
    I = Nij>0
    Nijk = Nijk[I,:]
    pijk = Nijk/Nij[I].reshape(Nij.size,1)
    LL = sum(Nijk[pijk>0]*np.log(pijk[pijk>0]))

    N=sink.size
    #qi=np.multiply.reduce(arity[1:])
    #out=sum(np.log(np.arange(1,ri)))
    #out-=sum([sum(np.log(np.arange(n+1,n+ri))) for n in Nij[I]])
    out = sum(np.log([(i)/(n+i) for i in range(1,ri) for n in Nij[I]]))
    out+=0.5*(sum(np.log(2*np.pi*Nijk[Nijk>0]))-sum(np.log(2*np.pi*Nij[I])))
    #out+=0*sum(((12.0*Nijk[Nijk>0])**-1))-sum(((12.0*Nij[I])**-1))
    #out-=0*sum((360.0*Nijk[Nijk>0])**-3)-sum((360.0*Nij[I])**-3)
    #out+=0.5*sum(np.log(pijk[pijk>0]))
    #print(out)

    #LL = -N*H
    #H(X|Y) = H(X,Y) - H(Y)
    #MI(X,Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y) 
    return -LL/N-out/N,0.0








