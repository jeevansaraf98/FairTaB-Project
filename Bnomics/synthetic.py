import numpy as np
import matplotlib.pyplot as plt
import itertools as itools


def random_adjmat(var_size , density = None , source_max = None):
    if density is None: density = np.random.randint( 1 , np.int(var_size*(var_size-1)/2) )
    if density>0 and density<1: density = np.int(density * var_size*(var_size - 1)*.5)
    tmp_vec = np.zeros( np.int(var_size*(var_size-1)/2) , dtype = np.int ) 
    tmp_vec[ 0 : density ] = 1
    np.random.shuffle( tmp_vec )
    adj_mat = np.zeros( (var_size,var_size) , dtype=np.int )
    for ind in range( var_size-1 ):
        adj_mat[ ind+1 , 0:ind+1 ] = tmp_vec[ np.int(ind*(ind+1)/2) : np.int((ind+1)*(ind+2)/2) ]
        if not(source_max is None):
            adj_mat[ ind+1 , np.where(adj_mat[ ind+1 , 0:ind+1 ] > 0)[0][source_max:] ] = 0
    return adj_mat


def factor_str( adj_mat ):
    out=''
    for node in range( adj_mat.shape[0] ):
        sources = adj_mat[ node ] > 0
        if np.sum( sources ) == 0: out+='{%s} ' %( node )
        else: out += '{%s%s} ' %( node , np.str(np.where(sources)[0]) )
    return out


def factor_indx( adj_mat ):
    out=[]
    for node in range( adj_mat.shape[0] ):
        sources = adj_mat[ node ] > 0
        if np.sum( sources ) == 0: out += [[ node ]]
        else: out += [[node] + np.where( sources )[0].tolist()]
    return np.array(out, dtype=object)



def joint_prob( node_prob , cond_prob , arity , adj_mat, return_states=False ):
    """ utility for random_dist function, returns an array of joint probabilities constructed for a
    given bundle of nodes """

    prod = np.zeros( adj_mat.shape[0] , dtype=object )
    factor = factor_indx( adj_mat ) 
    states = [np.arange(r) for r in arity]
    cartesian = np.stack(np.meshgrid(*states,indexing='ij'),-1).reshape(-1,len(states))
    dimension = np.array([len(factor[node]) for node in range(adj_mat.shape[0]) ])
    terms = node_prob[ dimension == 1].tolist() + cond_prob[ dimension > 1 ].tolist()
    out = np.zeros(len(cartesian))

    for k,indx in enumerate(cartesian):
        tmp=1
        s=''
        for i in range(adj_mat.shape[0]): 
            if dimension[i]>1:
                tmp_arity = np.concatenate(([1],arity[ factor[i][1:] ][:-1]))
                #tmp_arity[0]=1
                basis = np.multiply.accumulate(tmp_arity )
                #if i==6: print(factor[i],indx, indx[i],indx[ factor[i][1:]],\
                #    np.dot(indx[factor[i][1:] ],basis),basis,cond_prob[i].shape )
                #print(k,i,cond_prob[i].shape, arity[factor[i][1:]][:-1], ddd,indx[factor[i][1:]], basis)
                tmp *= cond_prob[i][ indx[i], np.dot(indx[ factor[i][1:] ], basis ) ]
                #s += str(cond_prob[i][ indx[i], np.dot(indx[ factor[i][1:] ], basis ) ])+' '

            else:
                tmp *= node_prob[i][ indx[i] ]
        out[k] = tmp
    if return_states is True:
        return out,cartesian
    else:
        return out


def random_dist( arity , adj_mat , a = 0, bb = 1, cc = 100):
    """ arity - array of node arities;
        adj_mat - adjacency matrix corresponding to the network structure;
        alpha - concentration parameter in the source dirichlet distribution;
        """

    node_prob = np.zeros( arity.size , dtype = object )
    cond_prob = np.zeros( arity.size,dtype = object )
    source_joint=np.zeros( arity.size,dtype = object )
    def topological_sort(adj_mat):
        n = adj_mat.shape[0]
        visited = [False] * n
        result = []

        def dfs(v):
            visited[v] = True
            for u in np.where(adj_mat[v])[0]:
                if not visited[u]:
                    dfs(u)
            result.append(v)

        for i in range(n):
            if not visited[i]:
                dfs(i)

        return result[::-1]

# In random_dist()
    for node in topological_sort(adj_mat.T):  # transpose to sort parents before children

        sources = adj_mat[ node ] > 0
        if np.sum( sources ) == 0:
            node_prob[ node ] = np.random.dirichlet( [1]*arity[node] )
            #np.random.shuffle(node_prob[ node ])
        else:
            ancestor_arity=np.prod(arity[sources])
            if a: alpha= np.random.gamma(bb,cc,size=arity[node])
            else: alpha=np.ones(arity[node])
            cond_prob[node] = np.random.dirichlet(alpha,size=ancestor_arity).T
            # Repeat sampling until dependency is enforced
            # while True:
            #     candidate_cpt = np.random.dirichlet(alpha, size=ancestor_arity).T
            #     if enforce_dependency(candidate_cpt):
            #         cond_prob[node] = candidate_cpt
            #         break

            #np.random.dirichlet(np.random.uniform(0,100,size=arity[node]),size=ancestor_arity).T
            #for i in range(arity[node]): np.random.shuffle(cond_prob[node][:,i])
            #E=np.eye(arity[node],ancestor_arity)
            #cond_prob[node] = (cond_prob[node])/np.sum(cond_prob[node],axis=0)
            source_joint[node] = joint_prob(\
                node_prob[sources],cond_prob[sources],\
                arity[sources], adj_mat[sources][:,sources]\
                )
            #print(factor_str(adj_mat[sources][:,sources]))
            node_prob[node]=np.dot(cond_prob[node],source_joint[node])
            #print('node\n %s\n' %np.sum(node_prob[node]))
    return [node_prob,cond_prob,source_joint]


def downstream_sampler( node_prob , cond_prob , arity , adj_mat , sample_size=100 ):
    samples = np.zeros( adj_mat.shape[0] , dtype = int )
    out=np.zeros((sample_size,adj_mat.shape[0]), dtype = int)
    for k in range(sample_size): 
        seed=np.random.uniform( size=arity.size )
        for node,row in enumerate(adj_mat):
            if np.sum(row) == 0:
                samples[node] = np.searchsorted( 
                np.add.accumulate(node_prob[node]) , seed[node] 
                )
                #print(node_prob[I[node]])
                #print(samples[I[node]])
            else:
                basis = np.multiply.accumulate( [1] + arity[row>0][:-1].tolist() )
                samples[node] = np.searchsorted(\
                     np.add.accumulate( cond_prob[ node ][ : , np.dot(samples[row>0],basis) ] ) ,\
                     seed[node]
                     )
                #print(cond_prob[I[node]][:,np.dot(samples[row>0],basis)] )
                #print(samples[I[node]])
        out[k]=samples
    return out

def downstream_sampler1( node_prob , cond_prob , arity , adj_mat , sample_size=100 ):
    #samples = np.zeros( adj_mat.shape[0] , dtype = np.int )
    samples=np.zeros((sample_size,adj_mat.shape[0]), dtype = np.int)
    #for k in range(sample_size): 
        #seed=np.random.rand( arity.size )
        #np.random.shuffle(seed)
    for node,row in enumerate(adj_mat):
        if np.sum(row) == 0:
            samples[:,node] = np.searchsorted( \
            np.add.accumulate(node_prob[node]) , np.random.rand(sample_size)
            )
            #print(node_prob[I[node]])
            #print(samples[I[node]])
        else:
            basis = np.multiply.accumulate( [1] + arity[row>0][:-1].tolist() )
            ancestors = np.dot(samples[:,row>0], basis)
            for state in np.unique(ancestors):
                samples[ ancestors == state , node ] = np.searchsorted(\
                    np.add.accumulate( cond_prob[ node ][ : , state ] ) ,\
                    np.random.rand(sum(ancestors == state))\
                 )
            #samples[node] = arange(arity[node])[np.random.multionmial(1,\
            #cond_prob[node][:,np.dot(samples[row>0],basis)])>0][0]
            #print(cond_prob[I[node]][:,np.dot(samples[row>0],basis)] )
            #print(samples[I[node]])
    return samples


def upstream_sampler( node_prob , cond_prob , arity , adj_mat , sample_size=100 ):
    samples = np.zeros( adj_mat.shape[0] , dtype = np.int )
    
    I = np.argsort( np.sum( adj_mat , axis = 0 ) ) # sort by number of sinks
    seed=np.random.rand( arity.size )
    for node,row in enumerate(adj_mat.T[I]):
        if np.sum(row) == 0:
            samples[I[node]] = np.searchsorted( 
            np.add.accumulate(node_prob[I[node]]) , seed[node] 
            )
            #print(node_prob[I[node]])
            #print(samples[I[node]])
        else:
            basis = np.multiply.accumulate( [1] + arity[row>0][1:].tolist() )
            samples[I[node]] = np.searchsorted(\
                 np.add.accumulate( cond_prob[ I[node] ][ : , np.dot(samples[row>0],basis) ] ) ,\
                 seed[node]
                 )
            #print(cond_prob[I[node]][:,np.dot(samples[row>0],basis)] )
            #print(samples[I[node]])
    return samples

# Continuous variables 

def conditional_sampl(x,mu,A):
    given1 = np.random.normal()
    given2 = np.linalg.solve( A[1:,1:] , x - mu[1:] - A[1:,0]*given1 )
    given2 = given2.tolist()
    given2.insert(0,given1)
    return mu[0] + np.dot(A[0,:],given2)

def make_A(sigmas,mat):
    Cov = np.outer(sigmas,sigmas.T)
    print(Cov)
    rho=0.4*(mat+mat.T)+np.diag(np.ones(sigmas.size))
    #rho=0.5*np.ones((sigmas.size,sigmas.size))+np.diag(np.ones(sigmas.size)*0.5)
    Cov = Cov*rho
    Cov = np.linalg.cholesky(Cov)
    #print(np.linalg.cholesky(Cov))
    print(Cov)
    return Cov

def cond_normal(mu,S,x,size=None):
    
    b=mu[-1]+np.dot(S[:-1,-1].T,np.dot(np.linalg.inv(S[:-1,:-1]),(x-mu[:-1])))
    print(b)
    A=S[-1,-1] - np.dot(S[:-1,-1].T,np.dot(np.linalg.inv(S[:-1,:-1]),S[:-1,-1]))
    print(A)
    return np.random.normal([b],[A],size)



def continuous_sampler(adj_mat, size):
    samples = np.zeros( adj_mat.shape[0] )
    #mu_node = np.random.randint(-1,1,size=adj_mat.shape[0])
    mu_node = np.random.normal(size=adj_mat.shape[0])
    print(mu_node)
    sigma_node = np.random.uniform(1,6,size=adj_mat.shape[0])
    ind = [[node]+np.where(row>0)[0].tolist() if sum(row)>0 else []
        for node,row in enumerate(adj_mat)]
    A = [ make_A(sigma_node[ii],adj_mat[ii,:][:,ii]) \
        if ii else [] for ii in ind]
    out=[]
    for n in range(size):
        for node,row in enumerate( adj_mat ):
            if np.sum(row) == 0:
                samples[node] =\
                np.random.normal(mu_node[node],scale=sigma_node[node])
            else:
                mu = mu_node[np.array([node]+np.where(row>0)[0].tolist())]
                samples[node] = conditional_sampl(samples[row>0],mu,A[node])
        out.append(samples)
        samples=np.zeros( samples.size)
    return np.array(out)


#Scatter plot distribution of random networks as a function 
def test(arity=np.array([3]*8), samples=10000, alpha=0):
    def tt(arity,a):
        A=random_adjmat(arity.size)
        rd=random_dist(arity,A,alpha)
        return rd[0][-1]

    z=np.array([tt(arity,1).tolist() for i in range(samples)])


#baricentric coordinate to cartesian conversion
    b=np.array([[0,np.sqrt(2)/2,np.sqrt(2)],[0,1,0]])
    zz=np.dot(b,z.T)

    plt.triplot(b[0,:],b[1,:])
    plt.scatter(zz[0,:],zz[1,:],s=.1)
