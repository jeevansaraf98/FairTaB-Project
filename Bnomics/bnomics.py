#//=============================================================
#//(c) 2011 Distributed under MIT-style license. 
#//(see LICENSE.txt or visit http://opensource.org/licenses/MIT)
#//=============================================================


import os
import textwrap as tw
import dutils
from ofunc import bdm, mdl, cpt, mu

try: 
    import oflib
    cmdl=oflib.cext().mdl
    cmu=oflib.cext().mu
except OSError: 
    print("Try running make")
    pass

from bnutils import bnet, np
import bnutils
from copy import deepcopy
import multiprocessing as mp
import time
from functools import reduce


class search:
    """ Methods for BN structure learning """

    def __init__(self, dt, ofunc=None, pscore=None, cache_size=None, target_nodes=None):
        """
        dt: dataset instance of dutils.dataset
        ofunc: objective function can be selected between MU, MDL or BDM. The defalut is MDL.
        If C extension is properly compiled fast MU and MDL options may also be availabe.
        For example, assuming that
        >> import bnomics as bn
        >> data = bn.dutils.loader("test_data")
        was already done, to select fast MDL scoring one should do
        >> out = bn.search(data, ofunc=bn.cmdl)
        the scoring method can be selected between bn.mu, bn.cmu, bn.mdl, bn.cmdl, bn.bdm.              
        """

        # default objective function
        if ofunc: self.objfunc = ofunc
        else: self.objfunc = mdl
        self.sensitivity=1
        
        # set some class variables
        self.data=dt.data
        self.arity=dt.arity
        self.variables=dt.variables
        self.target_nodes=target_nodes

        # make an empty network
        self.BN=bnet(self.variables)

        self.node_index=np.asarray([i for i,j in enumerate(self.variables)])

        # evaluate unconditional scores
        self.scores=np.array([self.objfunc(self.data[:,[i]],self.arity[[i]]) for i in self.node_index])
        self.net_score=np.sum(self.scores)

        # for score caching
        self.cache_size=self.arity.size
        if cache_size:
            self.cache_size=cache_size


        self.delta_cache=np.zeros((self.node_index.size,self.cache_size))
        self.delta_index=np.zeros((self.node_index.size,self.cache_size),dtype=int)
        self.delta_tmp=np.zeros(self.node_index.size)
        
        # primary score tracking arrays
        self.add_deltas=np.zeros(self.node_index.size)
        self.tmp_add = np.zeros(self.scores.shape)
        self.add_candidates=[[] for i in self.node_index]

        self.remove_deltas=np.zeros(self.node_index.size)
        self.remove_candidates=[[] for i in self.node_index]
        self.tmp_remove = np.zeros(self.scores.shape)

        self.edge_scores = [[] for i in self.node_index]

        # targeted or untargeted reconstruction
        if target_nodes:
            for target_node in target_nodes:
                self.add_score(target_node)
        else:
            for i in self.node_index:
                self.add_score(i)




    def d_score_serial(self,node):
        """ Score ancestor candidates as descendents 
        """
        a_set = self.BN.find_ancestors(node)
        for i in sefl.BN.pcandidates-a_set:
            new_score = self.objfunc(self.data[:,[i,node]] , self.arity[[i,node]])
            delta_tmp[i] = new_score - self.scores[cnode]
        self.delta_tmp[delta_tmp<0.0] = 0.0
        self.delta_cache[:,node] = delta_tmp
        delta_tmp[:] = 0.0

    def add_score(self,cnode):
        """ Simple first order search of an edge to add 
        """

        start=time.time()
        family=self.BN.pnodes[cnode]
        score=self.scores[cnode]
        old_score=score[0]
        delta_tmp=self.delta_tmp
        parent=[]
        dd = 0
        ss=0
        for i in self.BN.p_candidates(cnode):
            subset=[cnode]+family+[i]#np.sort(family+[i]).tolist()
            new_score=self.objfunc(self.data[:,subset],self.arity[subset])

            d = score[0] - new_score[0] - new_score[1]*self.sensitivity
            if d  > dd: 
                dd = d 
                #ss = new_score[1]
                parent=[i] 
                self.tmp_add[cnode] = new_score
                #cond = False
                
        self.add_deltas[cnode] = dd
        self.add_candidates[cnode]=parent
        #print(time.time()-start)

    def remove_score(self,cnode):
        """ Simple reverse search for an edge to remove 
        """
        family=self.BN.pnodes[cnode]
        score=self.scores[cnode]
        old_score=score[0]
        epsilon = 0.0
        dd=0.0
        ss=0.0 
        parent=[]
        for i in self.BN.pnodes[cnode]:
            indx = family.index(i)
            family.remove(i)
            subset = [cnode]+family
            drop_score = self.objfunc(self.data[:,subset],self.arity[subset])
            family.insert(indx,i)

            d = drop_score[0] - score[0] - score[1]*self.sensitivity
            if d <= dd : 
                dd = d
                ss=score[1]; epsilon=1.0; parent = [i]; 
                self.tmp_remove[cnode] = drop_score

        self.remove_candidates[cnode] = parent
        self.remove_deltas[cnode] = -dd+epsilon


    def add_score_old(self,cnode):
        """ Simple first order search of an edge to add 
        """

        start=time.time()
        family=self.BN.pnodes[cnode]
        score=self.scores[cnode]
        old_score=score
        delta_tmp=self.delta_tmp
        parent=[]
        for i in self.BN.p_candidates(cnode):
            subset=[cnode]+np.sort(family+[i]).tolist()
            new_score=self.objfunc(self.data[:,subset],self.arity[subset])
            if score<new_score: 
                score=new_score; parent=[i]; self.tmp_add[cnode] = new_score
        self.add_deltas[cnode]=score-old_score
        self.add_candidates[cnode]=parent
        #print(time.time()-start)



    def add_score_old(self,cnode):
        """ Simple first order search of an edge to add"""

        family=[cnode]+self.BN.pnodes[cnode]
        score=self.scores[cnode]
        delta_tmp=self.delta_tmp
        for i in self.BN.p_candidates(cnode):
            subset=family+[i]
            new_score=self.objfunc(self.data[:,subset],self.arity[subset])
            delta_tmp[i]=new_score-score
        self.delta_tmp[delta_tmp<0.0]=0.0
        self.delta_index[cnode,:]=np.argsort(delta_tmp)[-self.cache_size:]
        self.delta_cache[cnode,:]=delta_tmp[self.delta_index[cnode,:]]
        delta_tmp[delta_tmp>0]=0.0




    def remove_score_old(self,cnode):
        """ Simple reverse search for an edge to remove 
        """
        family=self.BN.pnodes[cnode]
        score=self.scores[cnode]
        old_score=score
        delta = 0
        parent=[]
        for i in self.BN.pnodes[cnode]:
            indx = family.index(i)
            family.remove(i)
            subset=[cnode]+family
            new_score=self.objfunc(self.data[:,subset],self.arity[subset])
            family.insert(indx,i)
            if new_score >= score: 
                score=new_score
                parent=[i]
                delta=1
                self.tmp_remove[cnode]=new_score
        self.remove_candidates[cnode]=parent
        self.remove_deltas[cnode]=score-old_score+delta


    def ancestor_pair(self,node):
        """ Find the best pair of ancestors for node.
        """
        ancestors=self.BN.pnodes[node]
        score=self.scores[node]
        tmp=0.0
        pair=[]
        for i in self.BN.p_candidates(node):
            for j in self.BN.p_candidates(node):
                if j>i:
                    subset=[node]+np.sort([i,j]+ancestors).tolist()
                    new_score=self.objfunc(self.data[:,subset],self.arity[subset])
                    delta=new_score-score
                    if delta>tmp:
                        tmp=delta*0.5
                        pair=[i,j]
            
        return pair,tmp
    
        

    def add_edge_and_sync(self,cnode,pnode):
        """ Add the edge to DAG, apply the induced constraints and update scores """

        #print('adding %s->%s' %(pnode,cnode))

        #self.scores[cnode]+=self.add_deltas[cnode]
        self.scores[cnode] = self.tmp_add[cnode]  #mods for mu


        self.BN.pconstraints[cnode].add(pnode)
        self.BN.pconstraints[pnode].add(cnode)
        self.BN.cnodes[pnode].append(cnode)
        self.BN.cnodes[pnode].sort()
        self.BN.pnodes[cnode].append(pnode)
        self.BN.pnodes[cnode].sort()
        self.add_score(cnode)

        # get ancestor and descendent sets and propagate necessary changes
        # through scores and constraints
        d_set=self.BN.find_descendents(cnode)
        a_set=self.BN.find_ancestors(pnode)
        for i in a_set:

            if d_set-self.BN.pconstraints[i]:
                self.BN.pconstraints[i]|=d_set
            self.add_score(i)





    def remove_edge_and_sync(self,cnode,pnode):
        """ Remove the edge from DAG, relax constraints and update the scores """

        #print('removing %s -> %s' %(pnode,cnode))

        # get ancestor and descendent sets
        d_set=self.BN.find_descendents(cnode)
        a_set=self.BN.find_ancestors(pnode)

        # modify the locality of the edge and update scores
        self.BN.pnodes[cnode].remove(pnode)
        self.BN.cnodes[pnode].remove(cnode)
        self.BN.pconstraints[cnode]-=set([pnode])
        family=[cnode]+self.BN.pnodes[cnode]
        self.scores[cnode]=self.tmp_remove[cnode] #self.objfunc(self.data[:,family],self.arity[family])
        self.add_score(cnode)
        self.remove_score(cnode)

        # propagate changes through constraints and scores
        for i in self.BN.pnodes[cnode]:
            a_set-=self.BN.find_ancestors(i)

        for i in a_set:
            update_constraints=self.BN.find_descendents(i)|set([i])|set(self.BN.pnodes[i])
            if self.BN.pconstraints[i]-update_constraints:
                self.BN.pconstraints[i] = update_constraints
            self.add_score(i)
  
            
    def ascent(self, max_iter=None):
        """ Data driven structure learning ascent.
            max_iter: the upper bound on the number of added to the BN edges
        """
        N = self.node_index.size
        if max_iter is None: max_iter=N*np.log(N)
        

        #best_deltas=np.amax(self.delta_cache,axis=1)
        #max_delta = np.max(best_deltas)

        # Initialize 
        #max_add = np.min(self.tmp_add[:,1])
        max_add = np.max(self.add_deltas)
        max_remove = np.max(self.remove_deltas)
        #max_remove = np.max(self.tmp_remove[:,1])

        #cnode=np.argmax(best_deltas)
        #pnode=np.argmax(self.delta_cache[cnode,:])


        cur_iter=0
        # iteratre until tolerances or max_iter is hit
        while (max_add>0.0 or max_remove>0.0) and (cur_iter<max_iter):

            # add an edge first and update conditionals
            if max_add>0.0:#max_remove:
                cnode = np.argmax(self.add_deltas)
                pnode = self.add_candidates[cnode][0]


                # check that the pair (cnode,pnode) doesn't create a cycle
                if pnode in self.BN.cnodes[cnode]:
                    print('cycle: %s->%s' %(pnode,cnode))
                    raise ValueError

                # add the egde and update the removal scores
                self.add_edge_and_sync(cnode,pnode)
                self.remove_score(cnode)

                # update conditionals 
                max_add = np.max(self.add_deltas)
                max_remove = np.max(self.remove_deltas)
    
            # remove an edge and update conditionals
            if max_remove>0.0:
                cnode=np.argmax(self.remove_deltas)
                pnode=self.remove_candidates[cnode][0]
                self.remove_edge_and_sync(cnode,pnode)

                # update conditionals after sync
                max_add = np.max(self.add_deltas)
                max_remove = np.max(self.remove_deltas)

                                                   
            cur_iter+=1

        return self.net_score,np.sum(self.scores)  



    def restarts(self, nrestarts=10, criterion=False, alpha=0.5):
        """ Ascent with stochastically perturbed restarts.

        nrestarts: integer number of restarts to perform;
        criterion: when provided the result of intrinsic_factorization() serves
        as a stoping criterion when searching for a predefined network;
        alpha: float in (0,1), the degree of perturbation, when small it aids in speeding up
        the search in large spaces by controlling/narrowing the perturbation
        window; """


        bn_search=self.ascent

        bn_search()
        tmpBN=deepcopy(self.BN)
        tmpscore=np.sum(self.scores) #MIR mod

        if criterion:
            res=[[1,tmpscore,tmpBN.intrinsic_factorization()==criterion]]
        else:
            res=[[1,tmpscore]]

        visited_BN=[]
        for restart in range(2,nrestarts+1):
            print("iteration %d" %restart)

            # Perturb the structure, check if this equivalence class was
            # visited before, and update scores.
            self.BN.perturb_net(alpha)
            ifac = self.BN.intrinsic_factorization()
            if ifac not in visited_BN: 
                visited_BN.append(ifac)
            self.score_net()
            for node in self.BN.node_index:
                self.add_score(node)
                self.remove_score(node)
            bn_search()
            current_score=np.sum(self.scores) #MIR mod

            # Check for improvement and, if criterion was set, check whether
            # equivalence class was reached.
            if current_score<tmpscore: #MIR mod
                print('found')
                tmpBN=deepcopy(self.BN)
                tmpscore=current_score
                if criterion:
                    equivalence = tmpBN.intrinsic_factorization()==criterion 
                    res.append([restart,tmpscore,equivalence]) 
                    if equivalence: break
                else:
                    res.append([restart,tmpscore])


        print("%s visited classes" %len(visited_BN))
        self.BN=deepcopy(tmpBN)
        self.score_net()
        return res




    def simple_search(self, max_edges=None, tol=10**-6):
        """ Simple data driven structure learning search.

            max_edges is the upper bound on the number of edges added to the BN.
        """

        # If max_edges is not provided try to add up to twice as many edges as
        # there are nodes 
        if max_edges is None: max_edges=2*self.node_index.size

        for i in self.node_index:
            self.add_score(i)
        best_deltas=np.amax(self.delta_cache,axis=1)
        cnode=np.argmax(best_deltas)
        pnode=np.argmax(self.delta_cache[cnode,:])
        cur_iter=0
        while np.max(best_deltas)>tol and cur_iter<max_edges:
            print( 'iteration %s' %cur_iter)
            print(best_deltas[cnode],cnode,self.delta_index[cnode,pnode])
            self.add_edge_and_sync(cnode,pnode)

            best_deltas=np.amax(self.delta_cache,axis=1)
            cnode=np.argmax(best_deltas)
            pnode=np.argmax(self.delta_cache[cnode,:])
            cur_iter+=1
        return self.net_score,np.sum(np.diag(self.scores))


    def score_net(self):
        """ Score the constructed BN """

        score=0
        for child in self.node_index:
            subset=[child]+self.BN.pnodes[child]
            self.scores[child]=self.objfunc(self.data[:,subset],self.arity[subset])
        self.net_score=np.sum(self.scores, axis=0)#[:,-1]) #MIR mod
        return self.net_score
    

    def score_edge(self, cnode, pnode):
        """ Score the edge given by (cnode,pnode) """
        bundle=[cnode]+self.BN.pnodes[cnode]
        
        score=self.scores[cnode]
        #score=self.objfunc(self.data[:,bundle],self.arity[bundle])
        try:
            bundle.remove(pnode)
        except ValueError:
            print(cnode,pnode)
        drop_score=self.objfunc(self.data[:,bundle],self.arity[bundle])

        return drop_score[0]-score[0]-score[1]*self.sensitivity

    def score_edges(self):
        self.edge_scores = [[self.score_edge(cnode,pnode) for pnode in self.BN.pnodes[cnode]] for cnode in self.BN.node_index]


    def get_cpt(self,subset):
        """ Helper for populate_prob function """

        Nijk,Nij,pstates,basis,H,C = cpt( self.data[:, subset], self.arity[subset] )
        unconditional=np.sum(Nijk,axis=0)
        vec=np.zeros(Nij.size)
        vec[Nij.nonzero()]=1/Nij[Nij.nonzero()]
        vec=np.diag(vec)
        return unconditional/np.sum(unconditional),np.dot( Nijk.T, vec),basis

    def populate_prob(self, node=None):
        """ Returns Markov Cover of a node with induced P(node : MC), 
            i.e. probability of node given its Markov Cover;

            node: integer index of the node in question; """
            
        if node is None: mc = self.BN
        else: mc=self.BN.markov_cover(node)
        mc.unconditional_p=np.zeros(mc.bsize,dtype=object)
        mc.conditional_p=np.zeros(mc.bsize,dtype=object)
        mc.basis=np.zeros(mc.bsize,dtype=object)
        print(mc.bsize) 
        for i in range(mc.bsize):

            subset = mc.global_index[[i] + mc.pnodes[i]]
            print(subset)
            mc.unconditional_p[i],mc.conditional_p[i],mc.basis[i]\
            = self.get_cpt(subset)
        if not(node is None): return mc
        
        


    def stats(self,node=None,filename='', return_Nijk=False):
        """ Print out P(node : ancestors) and some other useful data.

        node: integer index of the node of interest; 
        filename: string containing the name of the file to save to; 
        return_Nijk=False: if True the function will return the joint state
        counts Nijk for node_i and its ancestors where the column index k correspond to
        node states and row index j corresponds to ancestor joint states; """

        if node is None:
            nodes=self.BN.node_index
        else: nodes=[node]
        output=''
        for node in nodes:
            subset=[node]+self.BN.pnodes[node]
            Nijk,Nij,pstates,basis,H,C=cpt(self.data[:,subset],self.arity[subset])

            head='\t\t'.join(['%4s' %val  for val in np.unique(self.data[:,node])])
            output+='Node: '+str(self.BN.node_names[node])+':'+str(node)+'\n'
            tmp=''.join(['Ancestors: ',
                ' '.join(['('+str(self.BN.node_names[i])+':'+str(i)+')'
                for i in self.BN.pnodes[node]])])
            output+='\n'.join(tw.wrap(tmp))+'\n'
            tmp=''.join(['Descendants: ',
                    ' '.join(['('+str(self.BN.node_names[i])+':'+str(i)+')'
                for i in self.BN.cnodes[node]])])
            output+='\n'.join(tw.wrap(tmp))+'\n'
            output+='H=%f C=%f\n' %(H,C)
            output+='State\tCount\t'+head+'\n\n'
            for i,state in enumerate(pstates):
                output+=''.join([''.join(list(map(str,state))),\
                '\t%4s\t' %(Nij[i]),\
                '\t'.join(['%4s(%.2f)' %(val,val/Nij[i])\
                if Nij[i]>0 else '%4s(%.2f)' %(0,0.0) for val in Nijk[i,:] ]),'\n'])
            output+='\n\n'
        if filename:
            print(output,file=open(filename,'w'))
        else:
            print(output)
        if return_Nijk: return Nijk

    
        
        




    def dot(self, filename = "rendering", \
                    path = "", \
                    tol = 10**-16, \
                    cnode = None, \
                    radius = None,\
                    concentrate = False, \
                    edge_label = True, \
                    connected_only = True, \
                    logscale = False, \
                    return_scores=False):
        """ Create a dot file for the constructed BN.

            path:   optional string pointing to the location of the file to be created;
            tol:    minimum edge stregth required for the edge to be rendered;
            cnode:  optional node index to construct Markov neighborhood around;
            radius = None:          if int r is provided, constructs neighborhood of radius r around the node;
            edge_label = True:      if False, edge strengths labels are dropped;
            concentrate = False:    if True merges edges with common source together;
            logscale = False:       linear greyscale gradient, or logarithmic greyscale gradient to draw edges;
            connected_only = True:  render only the connected nodes, drop the rest;

        """

        BN=self.BN


        if not cnode is None:
            if radius is None:
                BN=self.BN.markov_cover(cnode)
                print( 'Markov neighborhood of %s' %self.BN.node_names[cnode])
            else:
                BN=self.BN.subnet_of_radius(cnode,radius)
                print("Subnet of radius %d around %s" \
                    %(radius,self.BN.node_names[cnode]) )
        
        # Index map for Markov Net scoring
        ind_map=[self.BN.node_names.index(i) for i in BN.node_names]

        edge_scores=[]
        edges=[]
        
        # Begin the dot string and set some global attributes 
        s='digraph G{\n ratio=fill;\n node [shape=box, style=rounded];\n\
            edge [penwidth=1];\n'
        if bool(concentrate):
            s+='concentrate=true;\n'

        
        if bool(connected_only):
            node_index=[i for i in BN.node_index if BN.pnodes[i]]
        else:
            node_index=BN.node_index
        

        #self.score_edges()
        for cnode in node_index:
            s+='"%s";\n' %BN.node_names[cnode]

            for pnode in BN.pnodes[cnode]:

                # In case BN is a Markov Neighborhood translate the edge
                # indices into their global equivalent and score. Otherwise the
                # map is an identity
                cnode_g=ind_map[cnode]
                pnode_g=ind_map[pnode] 

                edge_score = self.score_edge(cnode_g,pnode_g)

                print(edge_score)
                if edge_score>tol:
                    edge_scores.append(edge_score)
                    edges.append((pnode_g,cnode_g))

        
        # Generate grayscale color gradient for all the edges
        edge_scores = np.array(edge_scores)
        color_grad = np.ones(len(edge_scores))
        pos_scores = edge_scores[edge_scores>0]


        if bool(logscale):
            log_score_range=np.log(np.max(pos_scores))-np.log(np.min(pos_scores)) 
            color_grad[edge_scores > 0] = 0.9-(np.log(pos_scores)-np.log(np.min(pos_scores)))/log_score_range
            #color_grad=(1-np.array(np.log(np.abs(edge_scores)))/np.log(np.max(edge_scores)))
        else:
            score_range = np.max(pos_scores)
            color_grad[edge_scores>0]=(.9-np.array(pos_scores)/score_range)

        indx=np.argsort(edge_scores)
        
        # Add edges and style to the dot string
        for i,j in enumerate(edge_scores):


            if bool(edge_label):
                if j>0: 
                    col = '0 .2 %s' %(color_grad[i])
                    stl='[color="%s", fontcolor="%s", style=bold, label="%.4f" ]' %(col,col,j)
                else: stl='[color="0 0 0", style="dotted", arrowhead="empty" ]' 
            else:
                if j>0: stl='[color="0 0.2 %s"]' %(color_grad[i])   #style=bold ]'
                else: stl = '[color = "0.0 1.0 1.0"]'



            s+='"%s" -> "%s" %s;\n' %(BN.node_names[edges[i][0]],BN.node_names[edges[i][1]],stl)

        s+='}'

        # Write the dot string to file
        filestring = path+filename
        foo=open(filestring+".dot","w")
        foo.write(s)
        foo.close()
        os.system("dot -Tpdf "+filestring+".dot -o "+filestring+".pdf")

        if bool(return_scores): 
            return np.hstack([edge_scores.reshape((edge_scores.size,1)).astype(object),edges])
            
            

