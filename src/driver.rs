use crate::types::{StaticHFData, SCFResult};
use crate::integrals::integrals::{oneint, twoint};
use crate::{calc_nuc_rep, sym_matrix_packing, unpack_symmetric_matrix, orthonomalizer, get_occupation, get_rhf_energy, get_integral_index};
use crate::scf::do_scf;
use rayon::prelude::*;

pub fn calculate_single_point_energy(
    xyz: &[[f64; 3]], 
    static_data: &StaticHFData,
    p_matrix_guess: Option<&[f64]>,
    silent: bool 
) -> SCFResult { 
    
    let n_basis = static_data.n_basis;
    let ng = static_data.ng;

    let enuc = calc_nuc_rep(xyz, &static_data.z_nuc);

    let mut sab = vec![0.0; n_basis * n_basis];
    let mut tab = vec![0.0; n_basis * n_basis];
    let mut vab = vec![0.0; n_basis * n_basis];

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let r_a = static_data.centers[mu];
            let r_b = static_data.centers[nu];
            let alp_mu = &static_data.alphas[mu * ng..(mu + 1) * ng];
            let alp_nu = &static_data.alphas[nu * ng..(nu + 1) * ng];
            let ci_mu = &static_data.coeffs[mu * ng..(mu + 1) * ng];
            let cj_nu = &static_data.coeffs[nu * ng..(nu + 1) * ng];

            let (s, t, v) = oneint(&xyz.to_vec(), &static_data.z_nuc, &r_a, &r_b, alp_mu, alp_nu, ci_mu, cj_nu);
            sab[nu * n_basis + mu] = s;
            tab[nu * n_basis + mu] = t;
            vab[nu * n_basis + mu] = v;
        }
    }

    let packed_sab = sym_matrix_packing(&sab, n_basis);
    let packed_tab = sym_matrix_packing(&tab, n_basis);
    let packed_vab = sym_matrix_packing(&vab, n_basis);
    
    let mut h_core = vec![0.0; packed_tab.len()];
    for i in 0..h_core.len() {
        h_core[i] = packed_tab[i] + packed_vab[i];
    }
    let fock_matrix = h_core.clone();

    let s_inv_sqrt = orthonomalizer(&packed_sab, n_basis);
    let x_full = unpack_symmetric_matrix(&s_inv_sqrt, n_basis);
    let occupation = get_occupation(static_data.n_elec, n_basis, "R");
    
    let mut p_matrix = vec![0.0; n_basis * n_basis];
    let mut c_matrix = vec![0.0; n_basis * n_basis];
    let e_rhf_init;

    if let Some(guess) = p_matrix_guess {
        p_matrix.copy_from_slice(guess);
        let h_core_full = unpack_symmetric_matrix(&h_core, n_basis);
        e_rhf_init = get_rhf_energy(&h_core_full, &h_core_full, &p_matrix, n_basis, enuc); 
    } else {
        let c_prime = form_initial_guess_c_prime(&h_core, &x_full, n_basis);
        
        for i in 0..n_basis {
            for j in 0..n_basis {
                for k in 0..n_basis {
                    c_matrix[j * n_basis + i] += x_full[k * n_basis + i] * c_prime[j * n_basis + k];
                }
            }
        }
        
        for i in 0..n_basis {
            for j in 0..n_basis {
                for k in 0..n_basis {
                    p_matrix[j * n_basis + i] += occupation[k] * c_matrix[k * n_basis + i] * c_matrix[k * n_basis + j];
                }
            }
        }
        let h_core_full = unpack_symmetric_matrix(&h_core, n_basis);
        e_rhf_init = get_rhf_energy(&h_core_full, &h_core_full, &p_matrix, n_basis, enuc);
    }

    let num_pairs = (n_basis * (n_basis + 1)) / 2;
    let num_unique_integrals = (num_pairs * (num_pairs + 1)) / 2;
    let mut packed_2e_integrals = vec![0.0; num_unique_integrals];

    let mut tasks = Vec::with_capacity(num_unique_integrals);
    for i in 0..n_basis {
        for j in 0..=i {
            let ij = get_integral_index(i, j);
            for k in 0..n_basis {
                for l in 0..=k {
                    let kl = get_integral_index(k, l);
                    if ij >= kl {
                        tasks.push((i, j, k, l, get_integral_index(ij, kl)));
                    }
                }
            }
        }
    }


    let computed_integrals: Vec<(usize, f64)> = tasks.par_iter().map(|&(i, j, k, l, idx)| {
        let integral_val = twoint(
            &static_data.centers[i], &static_data.centers[j],
            &static_data.centers[k], &static_data.centers[l],
            &static_data.alphas[i * ng..(i + 1) * ng], &static_data.alphas[j * ng..(j + 1) * ng],
            &static_data.alphas[k * ng..(k + 1) * ng], &static_data.alphas[l * ng..(l + 1) * ng],
            &static_data.coeffs[i * ng..(i + 1) * ng], &static_data.coeffs[j * ng..(j + 1) * ng],
            &static_data.coeffs[k * ng..(k + 1) * ng], &static_data.coeffs[l * ng..(l + 1) * ng],
        );
        (idx, integral_val)
    }).collect();

    for (idx, val) in computed_integrals {
        packed_2e_integrals[idx] = val;
    }

    let mut e_kin = 0.0;
    let mut e_pot = 0.0;
    for i in 0..n_basis {
        for j in 0..n_basis {
            e_kin += p_matrix[j * n_basis + i] * tab[j * n_basis + i];
            e_pot += p_matrix[j * n_basis + i] * vab[j * n_basis + i];
        }
    }

    if !silent {
            println!("--- ENERGIES ---");
            println!("Nuclear Repulsion (E_NN): {}", enuc);
            println!("Kinetic Energy (T):       {}", e_kin);
            println!("Nuclear Attraction (V_Ne):{}", e_pot);
        }

    let (final_energy, c_matrix, eigval) = do_scf(
        &h_core, &fock_matrix, &mut p_matrix, n_basis, enuc, 
        e_rhf_init, &mut c_matrix, &x_full, &occupation, &packed_2e_integrals,
        false
    );
    let sab_full = crate::unpack_symmetric_matrix(&packed_sab, n_basis);
    
    SCFResult {
        energy: final_energy,
        p_matrix,
        c_matrix,
        eigval,
        ao_integrals: packed_2e_integrals,
        overlap: sab_full,
        h_core: unpack_symmetric_matrix(&h_core, n_basis),
    }
}

fn form_initial_guess_c_prime(h_core_packed: &[f64], x_full: &[f64], n_basis: usize) -> Vec<f64> {
    let fock_full = crate::unpack_symmetric_matrix(h_core_packed, n_basis);
    let mut temp = vec![0.0; n_basis * n_basis];
    let mut f_prime = vec![0.0; n_basis * n_basis];
    
    for i in 0..n_basis {
        for j in 0..n_basis {
            for k in 0..n_basis {
                temp[j*n_basis + i] += x_full[k*n_basis + i] * fock_full[j*n_basis + k];
            }
        }
    } 
    for i in 0..n_basis {
        for j in 0..n_basis {
            for k in 0..n_basis {
                f_prime[j*n_basis + i] += temp[k*n_basis + i] * x_full[j*n_basis + k];
            }
        }
    }

    let mut f_prime_packed = crate::sym_matrix_packing(&f_prime, n_basis);
    let mut eigval = vec![0.0; n_basis];
    let mut c_prime = vec![0.0; n_basis * n_basis]; 
    let mut work = vec![0.0; 3 * n_basis]; 
    let mut info = 0;

    unsafe {
        lapack::dspev(b'V', b'U', n_basis as i32, &mut f_prime_packed, &mut eigval, &mut c_prime, n_basis as i32, &mut work, &mut info);
    }
    if info != 0 { panic!("Error diagonalizing F': info = {}", info); }
    
    c_prime
}