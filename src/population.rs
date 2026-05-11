pub fn mulliken_pop_analysis(p_matrix: &[f64], sab_full: &[f64], basis_to_atom: &[usize], n_basis: usize, z_nuc: &[f64]) {
    let mut pop_per_atom = vec![0.0; basis_to_atom.iter().max().unwrap() + 1];

    for i in 0..n_basis {
        let atom_i = basis_to_atom[i];
        for j in 0..n_basis {
            // P_{ij} * S_{ij}
            let contrib = p_matrix[j * n_basis + i] * sab_full[j * n_basis + i];
            
            pop_per_atom[atom_i] += contrib;
        }
    }

    println!("\nMulliken Population Analysis:");
    println!("Total Population: {:.4}", pop_per_atom.iter().sum::<f64>());
    for (atom_idx, population) in pop_per_atom.iter().enumerate() {
        println!("Atom {}: Population = {:.4}", atom_idx + 1, population);
    }
    
    println!("Mulliken Charges:");
    for (atom_idx, population) in pop_per_atom.iter().enumerate() {
        let charge = z_nuc[atom_idx] - population;
        println!("Atom {}: Charge = {:.4}", atom_idx + 1, charge);
    }
}

pub fn _charge_density_calculation(
    p_matrix: &[f64], 
    alphas: &[f64], 
    coeffs: &[f64], 
    n_basis: usize, 
    ng: usize, 
    basis_to_atom: &[usize], 
    xyz: &[[f64; 3]],
    point: &[f64; 3] 
) -> f64 {
    let mut cd = 0.0; 
    let mut basis_val = vec![0.0; n_basis];

    for mu in 0..n_basis {
        let atom_idx = basis_to_atom[mu];
        let r_a = xyz[atom_idx]; 

        let dx = point[0] - r_a[0];
        let dy = point[1] - r_a[1];
        let dz = point[2] - r_a[2];
        let r2 = dx*dx + dy*dy + dz*dz;

        let mut val = 0.0;
        
        for i in 0..ng {
            let alpha = alphas[mu * ng + i];
            let coeff = coeffs[mu * ng + i];
            val += coeff * (-alpha * r2).exp();
        }
        basis_val[mu] = val;
    }

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            cd += p_matrix[nu * n_basis + mu] * basis_val[mu] * basis_val[nu];
        }
    }

    cd
}