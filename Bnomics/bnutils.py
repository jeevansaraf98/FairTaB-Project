#//=============================================================
#//(c) 2011 Distributed under MIT-style license. 
#//(see LICENSE.txt or visit http://opensource.org/licenses/MIT)
#//=============================================================


import numpy as np
import random
import csv
import os
from ofunc import cpt
import time

def bnetload(structure):
    """ structure can be a csv file or and adjacency matrix """
    if type(structure)==np.ndarray:
        bn=bnet(np.arange(structure.shape[1]))
        pnodes=[np.where(p>0)[0].tolist() for p in structure]
    else:
        lst=[row for row in csv.reader(open(structure,'r'))]
        bn=bnet([row[0] for row in lst])
        pnodes=[[int(i) for i in row[1:]] if len(row)>1 else [] for row in lst]
    for c,pset in enumerate(pnodes):
        for p in pset:
            bn.add_edge(c,p)
    return bn


class bnet:
    def __init__( self, node_names ):
        """ Initialize an empty network 
        """
        self.node_names = node_names
        self.bsize = len(node_names)

        self.node_index = np.arange(self.bsize)
        self.global_index = np.arange(self.bsize)
        self.pnodes = [[] for i in node_names]
        self.cnodes = [[] for i in node_names]
        #self.p_candidates=[set([i for i in self.node_index if i!=j]) for j in self.node_index]
        self.pcandidates=set(self.node_index)
        self.pconstraints=[set([i]) for i in self.node_index]
        
        self.pmax=self.bsize
        self.required_edges=[]
        self.forbidden_edges=[]

        self.unconditional_p=False
        self.conditional_p=False
        self.basis=False
        self.adj_mat=False
        self.adj_ind=False

    def __and__(self,net_b):
        intersection=bnet(self.node_names)
        
        for i,name in enumerate(intersection.node_names):
            [intersection.add_edge1(i,j) for j in\
            set(self.pnodes[self.node_names.index(name)])&\
            set(net_b.pnodes[net_b.node_names.index(name)])]

        return intersection

    def __sub__(self,net_b):
        delta=bnet(self.node_names)
        for i,name in enumerate(delta.node_names):
            [delta.add_edge(i,j) for j in\
            set(self.pnodes[self.node_names.index(name)])-\
            set(net_b.pnodes[net_b.node_names.index(name)])]

        return delta
    
    def adjacency_matrix(self):
        """ Populates lower triangular adjacency matrix adj_mat
            and a permutation index adj_ind. 
        """
        a_mat = np.zeros((self.bsize,self.bsize),dtype=int)
        for k in range(self.bsize): a_mat[k,self.pnodes[k]] = 1
        ind = np.argsort([len(self.find_ancestors(i)) for i in self.node_index])
        self.adj_mat = a_mat[ind][:,ind]
        self.adj_ind = ind

    def factor_indx( self ):
        """ Returns an array of [node, ancestors] lists from the adjacency
        matrix if it was populated.
        """
        if type(self.adj_mat) is np.ndarray: adj_mat = self.adj_mat
        else: self.adjacency_matrix()
        out=[]
        for node in range( adj_mat.shape[0] ):
            sources = adj_mat[ node ] > 0
            if np.sum( sources ) == 0: out += [[ node ]]
            else: out += [[node] + np.where( sources )[0].tolist()]
        return np.array(out)

    def joint_prob(self,states):
        """ Returns the joint probability for the joint state if populate_prob() was run
        """
        if (type(self.unconditional_p) is np.ndarray) and\
            (type(self.conditional_p) is np.ndarray):
            tmp=1
            for i,j in enumerate(states):
                indx=np.dot(states[self.pnodes[i]],self.basis[i][1:])
                tmp*=self.conditional_p[i][states[i],indx]
        return tmp


    def get_cp(self,node, data, arity):
        """ helper
        """
        subset = [node] + self.pnodes[node]
        Nijk,Nij,pstates,basis,H,C = cpt( data[:, subset], arity[subset] )
        unconditional=np.sum(Nijk,axis=0)
        vec=np.zeros(Nij.size)
        vec[Nij.nonzero()]=1/Nij[Nij.nonzero()]
        vec=np.diag(vec)
        return unconditional/np.sum(unconditional),np.dot( Nijk.T, vec),basis

    def intrinsic_factorization(self):
        """ Algebraic charachterization of an equivalence class of structures
            as described by Wong and Wu.       
        """
        numerator = ['.'.join(np.sort([i]+j).astype(str)) for i,j in enumerate(self.pnodes)]
        denominator = ['.'.join(np.sort(j).astype(str)) for i,j in enumerate(self.pnodes) if len(j)>0]
        common=np.intersect1d(numerator,denominator)
        for i in common: 
            numerator.remove(i)
            denominator.remove(i)
        return (set(np.sort(numerator)),set(np.sort(denominator)))

    def intrinsic_distance(self,other):
        """ Attempt to establish a distance between two structures
        """
        this=self.intrinsic_factorization()
        distance = lambda x,y: len(x.union(y) - x.intersection(y))
        return distance(this[0],other[0]) + distance(this[1],other[1])
        

    def p_candidates( self, cnode ):
        """ Returns possible ancestor candidates for a node
        """
        return self.pcandidates-self.pconstraints[cnode]

    def c_candidates( self, node ):
        """ Retruns possible descendent candidates for a node
        """
        return self.pcandidates-self.find_ancestors(node)

    def insert_node(self, node_id):
        """ Adds a new node to the existing network object
        """
        self.node_names.append(node_id)
        self.bsize+=1
        self.node_index=np.arange(self.bsize)
        self.pnodes.append([])
        self.cnodes.append([])
        self.pconstraints.append(set([self.node_index[-1]]))

    def remove_node( self, node_id = None ):
        """ Removes the specified node from the existing network object
            node_id - a string ID, or an integer index of the node
        """

        if node_id in self.node_names:
            indx = self.node_names.index( node_id )
        elif node_id in self.node_index:
            indx = node_id
        else:
            raise ValueError("Node not found")

        for ancestor in self.pnodes[ indx ]:
            self.remove_edge( indx, ancestor )

        for descendent in self.cnodes[ indx ]:
            self.remove_edge( descendent, indx )

 
        self.bsize-=1
        self.node_index = np.arange( self.bsize )
        self.pnodes.pop( indx )
        self.cnodes.pop( indx )
        self.pconstraints.pop( indx )
        self.pcandidates = set( self.node_index )
        self.node_names.pop( indx )
                



    def find_ancestors(self,node):
        """ Returns the set of all ancestors of a node
        """
        ancestors=set()
        def g(node,ancestors):
            if node not in ancestors:
                ancestors|=set([node])
                for p in self.pnodes[node]:
                    g(p,ancestors)
        g(node,ancestors)
        return ancestors

    def find_descendents(self,node):
        """ Returns the set of all descendents of a node
        """
        d_set=set()
        def g(node,d_set):
            if node not in d_set:
                d_set|=set([node])
                for c in self.cnodes[node]:
                    g(c,d_set)
        g(node,d_set)
        return d_set
    
    def add_edge(self,cnode,pnode):
        """ Inserts an edge given by (cnode,pnode)
        """
        self.pconstraints[cnode].add(pnode)
        self.pconstraints[pnode].add(cnode)
        self.cnodes[pnode].append(cnode)
        self.pnodes[cnode].append(pnode)
        d_set=self.find_descendents(cnode)
        a_set=self.find_ancestors(pnode)
        for i in a_set:
            self.pconstraints[i]|=d_set

    def add_edge1(self,cnode,pnode):
        """ Inserts the edge given by (cnode,pnode)
        """
        self.pconstraints[cnode].add(pnode)
        self.cnodes[pnode].append(cnode)
        self.pnodes[cnode].append(pnode)
        d_set=self.find_descendents(cnode)
        a_set=self.find_ancestors(pnode)
        for i in a_set:
            self.pconstraints[i]|=d_set


    def add_random_edge(self,add_function=None):
        """ Insert a random edge
        """
        if add_function: add_edge=add_function
        else: add_edge=self.add_edge
        #candidates=(np.array([len(i) for i in self.pnodes])<self.pmax)*\
        #   (np.array([len(j) for j in self.pconstraints])<self.bsize)
        candidates=np.array([len(self.p_candidates(i)) for i in self.node_index])
        if np.sum(candidates)>0:
            cnode=np.random.choice(np.where(candidates>0)[0])
            pnode=np.random.choice([i for i in self.p_candidates(cnode)])
            self.add_edge(cnode,pnode)
            print('add %s->%s' %(pnode,cnode))
        else:
            raise ValueError

    def remove_edge(self,cnode,pnode):
        """ Removes the edge given by (cnode,pnode)
        """
        d_set=self.find_descendents(cnode)
        a_set=self.find_ancestors(pnode)
        self.pnodes[cnode].remove(pnode)
        self.cnodes[pnode].remove(cnode)
        self.pconstraints[cnode]-=set([pnode])
        for i in a_set:
            self.pconstraints[i]=self.find_descendents(i)|\
                                set([i])|set(self.pnodes[i])



    def remove_random_edge(self,remove_function=None):
        """ Remove a random edge
        """
        ind=np.where(np.array([len(i) for i in self.pnodes])>0)[0]
        if ind.size>0:
            cnode=np.random.choice(ind)
            pnode=np.random.choice(self.pnodes[cnode])
            #print('removing %d -> %d' %(pnode,cnode))
            if remove_function:
                remove_function(cnode,pnode)
            else: 
                self.remove_edge(cnode,pnode)
        else: 
            raise ValueError
    
    def perturb_net_old(self,alpha=0.5):
        """ Stochastically perturbs the existing structure
            alpha: the degree of perturbation between 0 and 1, the smaller
            values narrow the window of perturbation speeding up the search for
            large networks.

        """
        n=self.bsize

        #max_add=n*(n-1)*0.5 - np.sum([len(aset) for aset in self.pnodes])

        #start=time.time()
        def get_add_edges():
            add_candidates = []
            for cnode in self.node_index:
                for pnode in self.p_candidates(cnode):
                    add_candidates.append((cnode,pnode))
            return add_candidates

        drop_candidates = []
        for cnode in self.node_index:
            for pnode in self.pnodes[cnode]:
                drop_candidates.append((cnode,pnode))
        
        add_candidates = get_add_edges()
        add_size = len(add_candidates)
        drop_size = len(drop_candidates)
        if add_size>1 : count = np.random.randint(np.log(add_size)+1)
        else: count=0
        s = 0
        while (add_size > 0) and (s < count):
            ind = np.random.randint(add_size)
            cnode,pnode = add_candidates[ind]
            self.add_edge(cnode,pnode)
            #print('add %s->%s' %(pnode,cnode))
            add_candidates=get_add_edges()
            add_size = len(add_candidates)
            #self.intrinsic_factorization()
            s+=1
        #print('add %s' %s)

        if drop_size * alpha >  1:
            ind = np.arange(drop_size)
            np.random.shuffle(ind)
            count = np.random.randint(1,int(drop_size*alpha)+1)
            for i in ind[: count]:
                cnode,pnode = drop_candidates[i]
                self.remove_edge(cnode,pnode)
                #print('drop %s->%s' %(pnode,cnode))
            #print('drop %s' %count)
        #print(time.time()-start)



    def perturb_net(self,alpha=0.5):
        """ Stochastically perturbs the existing structure
            alpha: the degree of perturbation between 0 and 1, the smaller
            values narrow the window of perturbation speeding up the search for
            large networks.

        """
        n=self.bsize

        #max_add=n*(n-1)*0.5 - np.sum([len(aset) for aset in self.pnodes])

        #start=time.time()
        def get_add_edges():
            add_candidates = []
            for cnode in self.node_index:
                for pnode in self.p_candidates(cnode):
                    add_candidates.append((cnode,pnode))
            return add_candidates

        drop_candidates = []
        for cnode in self.node_index:
            for pnode in self.pnodes[cnode]:
                drop_candidates.append((cnode,pnode))
        
        drop_size = len(drop_candidates)
        if drop_size * alpha >  1:
            ind = np.arange(drop_size)
            np.random.shuffle(ind)
            count = np.random.randint(1,int(drop_size*alpha)+1)
            for i in ind[: count]:
                cnode,pnode = drop_candidates[i]
                self.remove_edge(cnode,pnode)

        add_candidates = get_add_edges()
        add_size = len(add_candidates)

        if add_size>1 : count = np.random.randint(np.log(add_size)+1)
        else: count=0
        s = 0
        while (add_size > 0) and (s < count):
            ind = np.random.randint(add_size)
            cnode,pnode = add_candidates[ind]
            self.add_edge(cnode,pnode)
            #print('add %s->%s' %(pnode,cnode))
            add_candidates=get_add_edges()
            add_size = len(add_candidates)
            #self.intrinsic_factorization()
            s+=1
        #print('add %s' %s)


    def random_relaxation(self,nodes, remove_function=None):
        """ relaxation of structure within Markov Cover of a randomly selected node 
        """
        if remove_function:
            rm_edge=remove_function
        else:
            rm_edge=self.remove_edge
        for node in nodes:
            pnodes=np.random.choice(self.pnodes[node],size=np.random.randint(1,len(self.pnodes[node])+1),replace=0)
            for pnode in pnodes:
                rm_edge(node,pnode)

    def mc_relaxation(self,node, remove_function=None):
        """ relaxation of structure within Markov Cover of a randomly selected node 
        """
        if remove_function:
            rm_edge=remove_function
        else:
            rm_edge=self.remove_edge
        mc = [node] + [q for q,k in enumerate(self.pnodes) if node in k]
        np.random.shuffle(mc)
        for i in mc[:np.random.randint(1,len(mc)+1)]:
            if self.pnodes[i]:
                j=np.random.choice(self.pnodes[i])
                print(i,j)
                rm_edge(i,j)


    def make_random_net(self):
        """ Generate a random structure
        """
        adj_mat=np.tril(np.random.randint(0,2,size=(self.bsize,self.bsize)),-1)
        self.pnodes=[i.nonzero()[0].tolist() for i in adj_mat]
        self.cnodes=[i.nonzero()[0].tolist() for i in adj_mat.T]
        self.pconstraints=[set(np.arange(i,self.bsize)) for i in range(self.bsize)]
        

    def markov_cover(self,node):
        """ Returns a network object that corresponds to the Markov Cover of the
        provided node.
        """
        mnodes=[node]+self.pnodes[node]+self.cnodes[node]
        for i in self.cnodes[node]:
            mnodes+=self.pnodes[i]
        mnodes=np.unique(mnodes).tolist()
        markov=bnet([self.node_names[i] for i in mnodes])
        markov.global_index = np.array(mnodes)
        for i,name in enumerate(mnodes):
            for j in set(self.pnodes[name])&set(mnodes):
                markov.add_edge(i,mnodes.index(j))
        return markov

    def subnet_of_radius(self,node,radius=1):
        """ Returns a network object that corresponds to a subnetwork of the
        given radius around the provided node.
        """
        Mn=self.markov_cover(node)
        print( Mn.node_names)
        mnodes=[self.node_names.index(name) for name in Mn.node_names]
        print (mnodes)
        subnet_nodes=[]
        for r in range(radius):
            for i in mnodes:
                Mn=self.markov_cover(i)
                subnet_nodes+=[self.node_names.index(name) for name in Mn.node_names]
                print (Mn.node_names)
            subnet_nodes=np.unique(subnet_nodes).tolist()
            mnodes=[i for i in subnet_nodes]

        subnet_r=bnet([self.node_names[i] for i in subnet_nodes])
        for i,name in enumerate(subnet_nodes):
            for j in set(self.pnodes[name])&set(subnet_nodes):
                subnet_r.add_edge(i,subnet_nodes.index(j))

        return subnet_r

    def dot(self,filename='rendering',header=None):
        """ Render the existing structure into a file output.pdf
        """
        s='digraph G{\n concentrate=true;\n ratio=fill;\n'
        if header:
            s += header + ';\n'
        for child in self.node_index:
            s+='"%s";\n' %self.node_names[child]
            for parent in self.pnodes[child]:
                s+='"%s" -> "%s";\n' %(self.node_names[parent],self.node_names[child])
        s+='}'
        dotfile=open(filename+'.dot','w')
        dotfile.write(s)
        dotfile.close()
        os.system("dot -Tpdf "+filename+".dot -o "+filename+".pdf")


    def bnetsave(self,filename='bnstruct.csv'):
        """ Save the existing structure into a CSV file.
        """
        fout=open(filename,'w')
        csvwr=csv.writer(fout)
        for i,name in enumerate(self.node_names):
            csvwr.writerow([name]+self.pnodes[i])
        fout.close()


