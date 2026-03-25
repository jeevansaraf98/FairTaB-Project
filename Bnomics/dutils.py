#//=============================================================
#//(c) 2011 Distributed under MIT-style license. 
#//(see LICENSE.txt or visit http://opensource.org/licenses/MIT)
#//=============================================================


import numpy as np
import csv
import oflib

def loader(data,sep=',',skip_header=0, rowskip=[], colskip=[], axis=1, names=1, fromstring=0):
    """ Loads a data and returns a dataset instance.
        
        data: a filename, string of data, or an array of data;

        The file should contain variable names on the first row,
        or the first column. 
        The rest should be numerical sample values and nothing else!!!

        skip_header: number of lines to skip;

        rowkip: a list of row indecies to skip;;

        colskip: a list of column indecies to skip

        names: should be set to 0 if the variable names are missing or are 
        too cumbersome to be replaced by the variable indecies instead;
        
        axis: 0 for row-wise orientation of the data in the file, or 1 for
        column-wise orientation.

        
    """

    if (type(data)==str) and (fromstring==1):
        iterable=data.strip('\n').split('\n')
        content=np.array([i for i in csv.reader(iterable,delimiter=sep)])
    elif type(data)==np.ndarray:
        content=data
    else:
        csv_reader=csv.reader(open(data,'r'),delimiter=sep)
        for k in range(skip_header):
            next(csv_reader,None)
        content=np.array([i for i in csv_reader])
        #content=np.genfromtxt(filename,delimiter=sep,dtype=str)

    if rowskip:
        content=np.delete(content,rowskip,0)

    if colskip:
        content=np.delete(content,colskip,1)

    if axis==0: # if the file oriented column-wise
        content=content.T

    if names==0:
        variables=np.arange(content.shape[1]).tolist()
        offset=0
    else:
        variables=content[0].tolist()
        offset=1

    try:
        #if type(data)!=np.ndarray:
        content=np.array([conv_col(col) for col in
        content[offset:].T],dtype='object').T
        arity=np.array([np.unique(i).size for i in content.T])
        return dataset(variables,content,arity)
    except ValueError: 
        print( 'Data could not be loaded, failed converting to float.')
        return content



def check_type(col):
    """ helper for conv_col() 
    """
    return any(np.array([\
        isinstance(i,float) for i in col]))


def conv_col(col):
    """ Attempt to convert column from raw to integer data
    """
    try:
        if not check_type(col): 
            out=col.astype(int)
            unq=np.unique(out)
            if np.max(unq)!=unq.size-1:
                print('index correction')
                out=np.searchsorted(unq,out)
        else: raise ValueError
    except ValueError:
        print('Not an int')
        try: 
            out=col.astype(float)
        except ValueError:
            print('Falling back to searchsorted for conversion')
            out=np.searchsorted(np.unique(col),col)
    return out

def conv_row(row):
    """ row converter
    """
    out=[]
    for val in row:
        if type(val)==str:
            val=conv(val)
        if type(val)==str:
            print('Falling back to searchsorted method %s' %type(val))
            out=np.searchsorted(np.unique(row),row)
            break
        out.append(val)
    return out

def conv(val):
    try:
        if '.' in val: return float(val)
        else: return int(val)
    except ValueError:
        print ('Data could not be converted: %s' %val)
        return val    
    



class dataset:
    """ Basic methods for preprocessing data """

    def __init__(self,variables,data,arity):
        self.variables=variables
        self.data=data
        self.arity=arity

    def bin_quantize(self, variables=[], bins=3, min_const_samples_bin_size=1.0/3):
        """ Attempt max entropy binning quantization 

            variables:  a list or an array as in variables=[1,3,5]
            bins: the number of categories in the quantization
            min_const_samples_bin_size: determines the min count of
            constant samples that should be given a category of its own
        """
        self.edges=np.zeros((self.arity.size,bins+1))
        for i in variables:
            un_cnt=np.unique(self.data[:,i],return_counts=True)
            constvals=un_cnt[0][un_cnt[1]>self.data.shape[0]*min_const_samples_bin_size]
            mask=np.ones(self.data.shape[0],dtype=bool)
            cv_edges=[]
            if constvals.size>0:
                for j,cv in enumerate(constvals):
                    mask*=(self.data[:,i]!=cv)
                    cv_edges+=[np.min(self.data[self.data[:,i]==cv,i])]
                    self.data[self.data[:,i]==cv,i]=j

            size=np.sum(mask)/(bins-constvals.size)
            sorted_i=np.argsort(self.data[mask,i])
            edges=[ self.data[mask,i][sorted_i[int(size*num)-1]] \
                for num in range(1,bins-constvals.size) ]
            print(edges,constvals.size)
            self.edges[i]=cv_edges+[self.data[mask,i][sorted_i[0]]]+edges+[self.data[mask,i][sorted_i[-1]]]
            self.data[mask,i]=np.searchsorted(edges,self.data[mask,i])+constvals.size
            arity=len(edges)+1+constvals.size
            if arity==np.unique(self.data[:,i]).size : self.arity[i] = arity
            else : self.arity[i] = -1


    def range_quantize(self,variables=[],bins=3):
        """ Uniform range quantization
            
            variables: list of indecies of nodes to quantize
            bins: the number of categories in the quantization
        """
        self.edges=np.zeros((self.arity.size,bins-1))
        for i in variables:
            edges=np.linspace(min(self.data[:,i]),max(self.data[:,i]),bins+1)[1:-1]

            print(edges)
            self.edges[i]=np.unique(edges)
            self.data[:,i]=np.searchsorted(self.edges[i],self.data[:,i])
            self.arity[i]=np.unique(self.data[:,i]).size

        
    def requantize(self,variables=[]):
        """ Replaces the sample values with their index,
            useful if the data has inconsistencies.

            variables: list of indecies of troublesome nodes
        """

        for i in variables:
            un=np.unique(self.data[:,i]).tolist()
            for j in un:
                inds=np.where(self.data[:,i]==j)[0]
                self.data[inds,i]=un.index(j)
                

    def quantize_all(self, cond = 5, bins=3):
        """ Discretize everything with arity>5 without thinking. 
        """
        self.requantize(np.where(self.arity<=cond)[0])
        self.bin_quantize(np.where(self.arity>cond)[0],bins)
        self.data=self.data.astype(int)
