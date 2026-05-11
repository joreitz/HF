use std::process;
use crate::{sym_matrix_packing, unpack_symmetric_matrix, get_rhf_energy, get_integral_index};
use lapack::dspev;
extern crate intel_mkl_src;

pub fn do_scf(
    h_core: &[f64], 
    _fock_matrix: &[f64], 
    p_matrix: &mut [f64],      
    n_basis: usize, 
    enuc: f64, 
    mut e_rhf_init: f64,       
    c_matrix: &mut [f64],
    x_full: &[f64], 
    occupation: &[f64], 
    packed_2e_integrals: &[f64],
    silent: bool
) -> (f64, Vec<f64>, Vec<f64>) {
    
    if !silent {
        println!("\n########### SCF ###########");
        println!("Iter | Energy (Hartree) | ΔE");
    }
    let mut converged = false;
    let mut iter = 0;
    let max_iter = 100;
    
    let packed_len = n_basis * (n_basis + 1) / 2;
    let mut fock_packed = vec![0.0; packed_len];

    let h_core_full = unpack_symmetric_matrix(h_core, n_basis);
    let mut eigval = vec![0.0; n_basis]; 
    
    while !converged && iter < max_iter {
        iter += 1;
        
        for i in 0..n_basis {
            for j in 0..=i { 
                let ij = get_integral_index(i, j);
                fock_packed[ij] = h_core[ij];
                
                for k in 0..n_basis {
                    for l in 0..n_basis {
                        let kl = get_integral_index(k, l);
                        let ijkl = get_integral_index(ij, kl);
                        let ik = get_integral_index(i, k);
                        let jl = get_integral_index(j, l);
                        let ikjl = get_integral_index(ik, jl);
                        
                        fock_packed[ij] += p_matrix[l * n_basis + k] * (packed_2e_integrals[ijkl] - 0.5 * packed_2e_integrals[ikjl]);
                    }
                }
            }
        }
        
        let fock_full = unpack_symmetric_matrix(&fock_packed, n_basis);
        
        let e_rhf = get_rhf_energy(&h_core_full, &fock_full, &p_matrix, n_basis, enuc);

        let delta_e = (e_rhf - e_rhf_init).abs();
        
        if !silent {
            println!("{:>4} | {:>16.8} | {:.2e}", iter, e_rhf, delta_e);
        }
        
        if delta_e < 1e-9 && iter > 1 {
            converged = true;
            e_rhf_init = e_rhf;
            break;
        }
        e_rhf_init = e_rhf;

        let mut f_prime = vec![0.0; n_basis * n_basis];
        let mut temp = vec![0.0; n_basis * n_basis];
        
        // F' = X^T F X
        for i in 0..n_basis {
            for j in 0..n_basis {
                for k in 0..n_basis {
                    temp[j * n_basis + i] += x_full[k * n_basis + i] * fock_full[j * n_basis + k];
                }
            }
        } 
        for i in 0..n_basis {
            for j in 0..n_basis {
                for k in 0..n_basis {
                    f_prime[j * n_basis + i] += temp[k * n_basis + i] * x_full[j * n_basis + k];
                }
            }
        }

        let mut f_prime_packed = sym_matrix_packing(&f_prime, n_basis);

        let jobz = b'V'; 
        let uplo = b'U'; 
        let n = n_basis as i32;
        let mut c_prime = vec![0.0; n_basis * n_basis]; 
        let mut work = vec![0.0; 3 * n_basis]; 
        let mut info = 0; 

        unsafe {
            dspev(
                jobz, uplo, n,
                &mut f_prime_packed,
                &mut eigval, 
                &mut c_prime,
                n,
                &mut work,
                &mut info,
            );
        }
        if info != 0 {
            eprintln!("Error diagonalizing F': LAPACK info = {}", info);
            process::exit(1);
        }


        c_matrix.fill(0.0);
        for i in 0..n_basis {
            for j in 0..n_basis {
                for k in 0..n_basis {
                    c_matrix[j * n_basis + i] += x_full[k * n_basis + i] * c_prime[j * n_basis + k];
                }
            }
        }
        

        p_matrix.fill(0.0);
        for i in 0..n_basis {
            for j in 0..n_basis {
                for k in 0..n_basis {
                    p_matrix[j * n_basis + i] += occupation[k] * c_matrix[k * n_basis + i] * c_matrix[k * n_basis + j];
                }
            }
        }
    }
    let _fock_full = unpack_symmetric_matrix(&fock_packed, n_basis);

    if !silent {
        if converged {
            println!("\nSCF procedure converged successfully in {} iterations!", iter);
        } else {
            println!("\nWARNING: SCF did NOT converge after {} iterations!", max_iter);
        }
    
    }
    //write_2d_matrix(&fock_full, n_basis, n_basis, Some("Final Fock Matrix"), None, None).expect("Failed to write Fock matrix");
    
    return (e_rhf_init, c_matrix.to_vec(), eigval);
}
