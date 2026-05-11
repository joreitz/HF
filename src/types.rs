use std::fmt;

#[derive(Debug)]
pub struct System {
    pub n_atoms: f64,
    pub n_elec: f64,
    pub n_basis: f64,
    pub atoms: Vec<Atom>,
}

#[derive(Debug)]
pub struct Atom {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub z_nuc: f64,
    pub n_basis: f64,
    pub zetas: Vec<f64>,
}

#[derive(Debug)]
pub struct SCFResult {
    pub energy: f64,
    pub p_matrix: Vec<f64>,
    pub c_matrix: Vec<f64>,
    pub eigval: Vec<f64>,
    pub ao_integrals: Vec<f64>,
    pub overlap: Vec<f64>,
    pub h_core: Vec<f64>,
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  Coordinates:  X: {:>8.4} | Y: {:>8.4} | Z: {:>8.4}", self.x, self.y, self.z)?;
        writeln!(f, "  Z_nuc (Charge): {:>6.1}", self.z_nuc)?;
        writeln!(f, "  Basis Funcs:  {:>6}", self.n_basis)?;
        if !self.zetas.is_empty() {
            writeln!(f, "  Zeta Values:")?;
            for (j, zeta) in self.zetas.iter().enumerate() {
                writeln!(f, "    {:>2}. {:>10.4}", j + 1, zeta)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for System {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n=========================================================")?;
        writeln!(f, "                   HARTREE-FOCK SYSTEM                   ")?;
        writeln!(f, "=========================================================")?;
        writeln!(f, "Total Atoms:      {}", self.n_atoms)?;
        writeln!(f, "Total Electrons:  {}", self.n_elec)?;
        writeln!(f, "Total Basis Func: {}", self.n_basis)?;
        writeln!(f, "---------------------------------------------------------")?;
        for (i, atom) in self.atoms.iter().enumerate() {
            writeln!(f, "Atom {}:", i + 1)?;
            write!(f, "{}", atom)?; 
            writeln!(f, "---------------------------------------------------------")?;
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct StaticHFData {
    pub n_atoms: usize,
    pub n_basis: usize,
    pub n_elec: f64,
    pub ng: usize,
    pub z_nuc: Vec<f64>,
    pub alphas: Vec<f64>,
    pub coeffs: Vec<f64>,
    pub zetas: Vec<f64>,
    pub basis_to_atom: Vec<usize>,
    pub centers: Vec<[f64; 3]>,
}

impl StaticHFData {
    pub fn build(sys: &System, ng: usize) -> Self {
        let n_basis = sys.n_basis as usize;
        let mut basis_to_atom = Vec::with_capacity(n_basis);
        let mut z_nuc = Vec::with_capacity(sys.n_atoms as usize);
        
        let mut centers = Vec::with_capacity(n_basis);

        for (atom_idx, atom) in sys.atoms.iter().enumerate() {
            z_nuc.push(atom.z_nuc);
            for _ in 0..(atom.n_basis as usize) {
                basis_to_atom.push(atom_idx);
                centers.push([atom.x, atom.y, atom.z]); 
            }
        }

        let mut alphas = vec![0.0; n_basis * ng];
        let mut coeffs = vec![0.0; n_basis * ng];
        let all_zetas: Vec<f64> = sys.atoms.iter().flat_map(|a| a.zetas.iter().copied()).collect();

        for mu in 0..n_basis {
            let zeta = all_zetas[mu];  
            let start = mu * ng;
            let end = (mu + 1) * ng;
            let (alpha_vec, coeff_vec) = crate::slater::slater::expand_slater(ng, zeta)
                .expect("Slater expansion failed");
            alphas[start..end].copy_from_slice(&alpha_vec);
            coeffs[start..end].copy_from_slice(&coeff_vec);
        }

        Self {
            n_atoms: sys.n_atoms as usize,
            n_basis,
            n_elec: sys.n_elec,
            ng,
            z_nuc,
            zetas: all_zetas,
            alphas,
            coeffs,
            basis_to_atom,
            centers, 
        }
    }
}