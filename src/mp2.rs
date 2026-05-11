use crate::get_integral_index;
use crate::types::{StaticHFData, SCFResult};
use std::time::Instant;

fn get_ao_integral(mu: usize, nu: usize, lambda: usize, sigma: usize, packed_2e: &[f64]) -> f64 {
    let ij = get_integral_index(mu, nu);
    let kl = get_integral_index(lambda, sigma);
    let idx = get_integral_index(ij, kl);
    packed_2e[idx]
}

pub fn do_mp2_cycle(scf_res: &SCFResult, static_data: &StaticHFData) -> f64 {
    let n_basis = static_data.n_basis;
    let n_occ = (static_data.n_elec / 2.0).floor() as usize; 
    let n_vir = n_basis - n_occ; 

    println!("\n=========== MP2 CORRECTION ===========");
    println!("Basis Functions: {}", n_basis);
    println!("Occupied MOs:    {}", n_occ);
    println!("Virtual MOs:     {}", n_vir);
    
    let start_time = Instant::now();

    let idx1 = |mu, nu, lam, b| ((mu * n_basis + nu) * n_basis + lam) * n_vir + b;
    let idx2 = |mu, nu, j, b|   ((mu * n_basis + nu) * n_occ + j) * n_vir + b;
    let idx3 = |mu, a, j, b|    ((mu * n_vir + a) * n_occ + j) * n_vir + b;
    let idx4 = |i, a, j, b|     ((i * n_vir + a) * n_occ + j) * n_vir + b;

    let c = &scf_res.c_matrix;
    
    // 1.
    print!("Step 1/4: Quarter-Transformation... ");
    let mut temp1 = vec![0.0; n_basis * n_basis * n_basis * n_vir];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for lam in 0..n_basis {
                for b in 0..n_vir {
                    let mut sum = 0.0;
                    for sig in 0..n_basis {
                        let c_sig_b = c[(b + n_occ) * n_basis + sig]; // b + nocc für virt
                            sum += c_sig_b * get_ao_integral(mu, nu, lam, sig, &scf_res.ao_integrals);
                    }
                    temp1[idx1(mu, nu, lam, b)] = sum;
                }
            }
        }
    }
    println!("Done.");

    // 2.
    print!("Step 2/4: Half-Transformation...    ");
    let mut temp2 = vec![0.0; n_basis * n_basis * n_occ * n_vir];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut sum = 0.0;
                    for lam in 0..n_basis {
                        let c_lam_j = c[j * n_basis + lam];
                            sum += c_lam_j * temp1[idx1(mu, nu, lam, b)];
                    }
                    temp2[idx2(mu, nu, j, b)] = sum;
                }
            }
        }
    }
    println!("Done.");

    // 3.
    print!("Step 3/4: Three-Quarter-Transform...");
    let mut temp3 = vec![0.0; n_basis * n_vir * n_occ * n_vir];
    for mu in 0..n_basis {
        for a in 0..n_vir {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut sum = 0.0;
                    for nu in 0..n_basis {
                        let c_nu_a = c[(a + n_occ) * n_basis + nu];
                            sum += c_nu_a * temp2[idx2(mu, nu, j, b)];
                    }
                    temp3[idx3(mu, a, j, b)] = sum;
                }
            }
        }
    }
    println!("Done.");

    // 4.
    print!("Step 4/4: Full MO-Transformation... ");
    let mut mo_ints = vec![0.0; n_occ * n_vir * n_occ * n_vir];
    for i in 0..n_occ {
        for a in 0..n_vir {
            for j in 0..n_occ {
                for b in 0..n_vir {
                    let mut sum = 0.0;
                    for mu in 0..n_basis {
                        let c_mu_i = c[i * n_basis + mu];
                        sum += c_mu_i * temp3[idx3(mu, a, j, b)];
                    }
                    mo_ints[idx4(i, a, j, b)] = sum;
                }
            }
        }
    }
    println!("Done.");

    // MP2 

    let mut e_mp2 = 0.0;
    let eps = &scf_res.eigval;

    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in 0..n_vir {
                for b in 0..n_vir {
                    let ia_jb = mo_ints[idx4(i, a, j, b)];
                    let ib_ja = mo_ints[idx4(i, b, j, a)];
                    
                    let denom = eps[i] + eps[j] - eps[a + n_occ] - eps[b + n_occ];
                    
                    let nom = ia_jb * (2.0 * ia_jb - ib_ja);
                    
                    e_mp2 += nom / denom;
                }
            }
        }
    }
    
    let duration = start_time.elapsed();
    println!("MP2 Transformation & Eval Time: {:.3} s", duration.as_secs_f64());
    println!("=============================================");
    
    e_mp2
}