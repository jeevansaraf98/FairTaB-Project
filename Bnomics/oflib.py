#//=============================================================
#//(c) 2011 Distributed under MIT-style license. 
#//(see LICENSE.txt or visit http://opensource.org/licenses/MIT)
#//=============================================================


import os

import numpy as np

class cext:
    def __init__(self):
        """ ctypes interface for C++ objective function implementation
        """
        _path=os.path.dirname(os.path.realpath(__file__))
        self.lib = np.ctypeslib.load_library('ofext',_path)

        #self.lib.mdl.restype=np.ctypeslib.ctypes.c_double
        type_data=np.ctypeslib.ndpointer(dtype=np.int32,ndim=2,flags='C_CONTIGUOUS')
        type_arity=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        type_out=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.mdl.argtypes=\
            [type_data,\
            np.ctypeslib.ctypes.c_int,\
            type_arity,\
            np.ctypeslib.ctypes.c_int,\
            np.ctypeslib.ctypes.c_int,\
            type_out]

        #self.lib.mu.restype=np.ctypeslib.
        type_data=np.ctypeslib.ndpointer(dtype=np.int32,ndim=2,flags='C_CONTIGUOUS')
        type_arity=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        type_out=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.mu.argtypes=\
            [type_data,\
            np.ctypeslib.ctypes.c_int,\
            type_arity,\
            np.ctypeslib.ctypes.c_int,\
            type_out]

        

    def mu( self, data, arity ):
        m,n = data.shape
        data = np.require(data, np.int32, 'C')
        arity = np.require(arity ,np.int32, 'C')
        out = np.zeros(2)
        self.lib.mu(data, m, arity, n, out)
        return out


    def mdl(self,data,arity,complexity=1):
        """ cpp mdl objective function
            complexity can be 0 for AIC or 1 for BIC"""
        m,n=data.shape
        data=np.require(data,np.int32,'C')
        arity=np.require(arity,np.int32,'C')
        out = np.zeros(2)
        self.lib.mdl(data,m,arity,n,complexity,out)
        return out
        
#    def aic(self,a,b):
#        """ cpp mdl (AIC) objective function
#            a - data
#            b - arity
#        """
#        cc=0
#        m,n=a.shape
#        a=np.require(a,np.int32,'C')
#        b=np.require(b,np.int32,'C')
#        out = np.zeros(2)
#        self.lib.mdl(a,m,b,n,cc,out)
#        return out
#
#
#    def mdl(self,a,b):
#        """ cpp mdl (BIC) objective function
#            a - data
#            b - arity
#        """
#        cc=1
#        m,n=a.shape
#        a=np.require(a,np.int32,'C')
#        b=np.require(b,np.int32,'C')
#        out=zeros(2)
#        self.lib.mdl(a,m,b,n,cc,out)
#        return out



