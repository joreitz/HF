from pyscf import gto, scf, tools

mol = gto.M(
    atom = 'He 0 0 0; He 0 0 5.6',
    basis = 'cc-pvdz', 
    unit = 'Bohr',
    symmetry = False
)

mf = scf.RHF(mol)
mf.kernel()

tools.fcidump.from_scf(mf, 'FCIDUMP_He2.txt')
print("FCIDUMP erfolgreich erstellt!")