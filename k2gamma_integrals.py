#!/usr/bin/env python

'''
Access AO integrals in PBC code
'''

from pyscf.pbc import gto, df, tools
import numpy as np
from pyscf import lib

def get_phase(cell, kpts, kmesh=None):
    '''
    The unitary transformation that transforms the supercell basis k-mesh
    adapted basis.
    '''

    latt_vec = cell.lattice_vectors()
    if kmesh is None:
        # Guess kmesh
        scaled_k = cell.get_scaled_kpts(kpts).round(8)
        kmesh = (len(np.unique(scaled_k[:,0])),
                 len(np.unique(scaled_k[:,1])),
                 len(np.unique(scaled_k[:,2])))

    R_rel_a = np.arange(kmesh[0])
    R_rel_b = np.arange(kmesh[1])
    R_rel_c = np.arange(kmesh[2])
    R_vec_rel = lib.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_vec_abs = np.einsum('nu, uv -> nv', R_vec_rel, latt_vec)

    NR = len(R_vec_abs)
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))
    phase /= np.sqrt(NR)  # normalization in supercell

    # R_rel_mesh has to be construct exactly same to the Ts in super_cell function
    scell = tools.super_cell(cell, kmesh)
    return scell, phase

cell = gto.Cell()
cell.atom='''
H 0.000000000000   0.000000000000   0.000000000000
H 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 4
cell.build()

kpts = cell.make_kpts([3,1,1])
nkpts = len(kpts)

# Get overlap matrix and kinetic matrix
#overlap = cell.pbc_intor('cint1e_ovlp_sph',kpts=kpts)
#kinetic = cell.pbc_intor('cint1e_kin_sph',kpts=kpts)
#print 'overlap shape'
#print len(overlap),overlap[0].shape

# Get eri from 3-index integrals at Gamma point
mydf = df.GDF(cell,kpts)
#eri = mydf.get_eri(compact=False)
#print('ERI shape at gamma',eri.shape)

# Get eri for all k points
'''
kconserv = tools.get_kconserv(cell, kpts)
for ki in range(nkpts):
    for kj in range(nkpts):
        for kk in range(nkpts):
            kl = kconserv[ki,kj,kk]
            eri = mydf.get_eri([kpts[i] for i in (ki,kj,kk,kl)])
'''
#
# Using .sr_loop method to access the 3-index tensor of gaussian density
# fitting (GDF) for arbitrary k-points
#
nao = cell.nao_nr()
eri_3d_kpts = []
for i, kpti in enumerate(kpts):
    eri_3d_kpts.append([])
    for j, kptj in enumerate(kpts):
        eri_3d = []
        for LpqR, LpqI, sign in mydf.sr_loop([kpti,kptj], compact=False):
            eri_3d.append(LpqR+LpqI*1j)
        eri_3d = np.vstack(eri_3d).reshape(-1,nao,nao)
        eri_3d_kpts[i].append(eri_3d)
print 'eri_3d shape'
print len(eri_3d_kpts),len(eri_3d_kpts[0]),eri_3d.shape

# Test 2-e integrals
#eri = np.einsum('Lpq,Lrs->pqrs', eri_3d_kpts[0][2], eri_3d_kpts[2][0])
#print(abs(eri - mydf.get_eri([kpts[0],kpts[2],kpts[2],kpts[0]]).reshape([nao]*4)).max())

print("Transform k-point integrals to supercell integral")
scell, phase = get_phase(cell, kpts)
NR, Nk = phase.shape
s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
s = scell.pbc_intor('int1e_ovlp')
s1 = np.einsum('Rk,kij,Sk->RiSj', phase, s_k, phase.conj())
print(abs(s-s1.reshape(s.shape)).max())

eri_3d_kpts = np.asarray(eri_3d_kpts)
eri_3d_k2gamma = np.einsum('Rk,kmLpq,Sm->LRpSq',phase,eri_3d_kpts,phase.conj())
eri_3d_k2gamma = eri_3d_k2gamma.reshape()
print eri_3d_k2gamma.shape

mydf_scell = df.GDF(scell)
eri_scell = mydf_scell.get_eri(compact=False).reshape([nao*nkpts]*4)
print eri_scell.shape
