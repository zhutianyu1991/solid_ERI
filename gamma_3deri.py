#!/usr/bin/env python

'''
Access AO integrals (3D ERI) in PBC code
'''


from pyscf.pbc import gto, df
import numpy

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 4,
)

#
# Using .sr_loop method to access the 3-index ERI tensor (L|pq) at gamma point.
#
nao = cell.nao_nr()
mydf = df.GDF(cell)
eri_3d = numpy.vstack([LpqR.copy() for LpqR, LpqI, sign in mydf.sr_loop(compact=False)]).reshape(-1,nao,nao)
print eri_3d.shape
