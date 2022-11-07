#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse
    import psi4
    import IPython
    import numpy as np
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Do an SCF calculation using Psi4 and save the density matrix and basis set information.')
    parser.add_argument('molecule', metavar='MOLECULE', help='xyz file')
    parser.add_argument('basis', metavar='BASIS', help='basis set')
    parser.add_argument('-o', '--output', metavar='OUTPUT', help='output file stem (json and npy)')

    args = parser.parse_args()

    psi4.core.set_global_option('PUREAM', False)
    mol = psi4.geometry(open(args.molecule).read())
    psi4.set_options({'basis': args.basis})

    E, wfn = psi4.energy('scf', return_wfn=True)

    Da = wfn.Da()
    Db = wfn.Db()
    Da.add(Db)
    Dt = Da

    npdt = Dt.to_array(dense=True)
    if args.output:
        filestem = args.output
    else:
        filestem = Path(args.molecule).stem + '-' + Path(args.basis).stem
    np.save(filestem + '.npy', npdt)

    sad_basis_list = psi4.core.BasisSet.build(wfn.molecule(), "ORBITAL",
        psi4.core.get_global_option("BASIS"), puream=wfn.basisset().has_puream(),
                                         return_atomlist=True)
    sad_fitting_list = psi4.core.BasisSet.build(wfn.molecule(), "DF_BASIS_SAD",
        psi4.core.get_option("SCF", "DF_BASIS_SAD"), puream=wfn.basisset().has_puream(),
                                           return_atomlist=True)
    ndocc = wfn.nalpha()
    nbeta = wfn.nbeta()
    SAD = psi4.core.SADGuess.build_SAD(wfn.basisset(), sad_basis_list)

    SAD.set_atomic_fit_bases(sad_fitting_list)
    SAD.compute_guess()
    SAD_Dt = SAD.Da()
    SAD_Dt.add(SAD.Db())
    npSAD_Dt = SAD_Dt.to_array(dense=True)
    np.save(filestem + '-SAD.npy', npSAD_Dt)

    b = wfn.basisset()
    nshell = b.nshell()
    nprimitive = b.nprimitive()
    nbf = b.nbf()
    natoms = mol.natom()

    atom_coords = [(mol.x(i), mol.y(i), mol.z(i)) for i in range(natoms)]
    atom_Z = [mol.Z(i) for i in range(natoms)]
    shells = []
    
    for shell in range(nshell):
        gs = b.shell(shell)
        shell_dict = {
            'coords': atom_coords[gs.ncenter],
            'Z': atom_Z[gs.ncenter],
            'am': gs.am,
            'nprim': gs.nprimitive,
            'coefs': [gs.coef(p) for p in range(gs.nprimitive)],
            'original_coefs': [gs.original_coef(p) for p in range(gs.nprimitive)],
            'exps': [gs.exp(p) for p in range(gs.nprimitive)]
        }
        shells.append(shell_dict)
    with open(filestem + '.json', 'w') as f:
        json.dump(shells, f, indent=1)