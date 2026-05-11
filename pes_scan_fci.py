import numpy as np
from pyscf import gto, scf, mp, tools
import subprocess
import re

# Wir scannen von 3.0 bis 10.0 Bohr in 0.2er Schritten
distances = np.arange(3.0, 10.2, 0.2)

print("Starte ultimativen PES-Scan mit PySCF und nativen Rust-FCI...")

with open('plotrhf_he2.dat', 'w') as frhf, \
     open('plotrmp2_he2.dat', 'w') as fmp2, \
     open('plotfci_he2.dat', 'w') as ffci:

    for d in distances:
        # 1. Molekül für diese Distanz bauen
        mol = gto.M(atom=f'He 0 0 0; He 0 0 {d}', basis='aug-cc-pvdz', unit='Bohr', symmetry=False)

        # 2. PySCF: RHF und MP2 ausrechnen
        mf = scf.RHF(mol)
        mf.verbose = 0 # Unterdrückt den Spam in der Konsole
        e_rhf = mf.kernel()

        mymp2 = mp.MP2(mf)
        mymp2.verbose = 0
        e_mp2, _ = mymp2.kernel()
        e_mp2_total = e_rhf + e_mp2

        # 3. FCIDUMP für Rust generieren
        tools.fcidump.from_scf(mf, 'FCIDUMP_He2.txt', tol=1e-15)

        # 4. DEIN RUST PROGRAMM AUFRUFEN!
        # Wichtig: Da wir in WSL sind, rufen wir die Windows-Exe von Cargo auf
        rust_command = ["cargo.exe", "run", "--release", "FCIDUMP_He2.txt", "0", "fcidump"]
        
        result = subprocess.run(rust_command, capture_output=True, text=True)

        # 5. FCI-Energie aus dem Rust-Konsolen-Output fischen (mit Regex)
        match = re.search(r'Total FCI Energy\s*:\s*([-.\d]+)', result.stdout)
        
        if match:
            e_fci = float(match.group(1))
        else:
            print(f"FEHLER bei {d} Bohr: Konnte Rust-Output nicht lesen!")
            print(result.stdout)
            e_fci = 0.0

        # 6. Alles in deine .dat Dateien schreiben
        frhf.write(f"{d:.2f}  {e_rhf:.10f}\n")
        fmp2.write(f"{d:.2f}  {e_mp2_total:.10f}\n")
        ffci.write(f"{d:.2f}  {e_fci:.10f}\n")
        
        print(f"Distanz: {d:.2f} Bohr | RHF: {e_rhf:.7f} | MP2: {e_mp2_total:.7f} | FCI: {e_fci:.7f}")

print("\nScan abgeschlossen! Du kannst jetzt dein Matplotlib-Skript ausführen.")