use std::collections::HashMap;
use rayon::prelude::*;
use crate::get_integral_index;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Determinant {
    pub alpha: u64,
    pub beta: u64,
}

#[inline]
fn get_phase(state: u64, idx: usize) -> f64 {
    if (state & ((1u64 << idx) - 1)).count_ones() % 2 == 1 { -1.0 } else { 1.0 }
}

// combining every alpha string with every beta string, alls determinants can be generated. Bitstrings for efficient storage and manipulation
pub fn generate_combinations (n_orbitals: usize, n_elec: usize) -> Vec<u64> {
    let mut combs = Vec::new();

    if n_elec == 0 {
        combs.push(0);
        return combs;
    }

    let mut state: u64 = (1 << n_elec) - 1; // Start with the first combination
    let limit = 1 << n_orbitals;

    while state < limit {
        combs.push(state);
        let c = state & state.wrapping_neg();
        let r = state + c;
        state = (((r ^ state) >> 2) / c) | r;
    }
    combs
}

pub fn build_fci_space(n_orbitals: usize, n_elec_alpha: usize, n_elec_beta: usize) -> Vec<Determinant> {
    let alpha_strings = generate_combinations(n_orbitals, n_elec_alpha);
    let beta_strings = generate_combinations(n_orbitals, n_elec_beta);
    
    let mut determinants = Vec::with_capacity(alpha_strings.len() * beta_strings.len());
    
    for &alpha in &alpha_strings {
        for &beta in &beta_strings {
            determinants.push(Determinant { alpha, beta });
        }
    }
    
    determinants
}

/// a^dagger_p a_q on bit string
/// new bit and phase factor (+1.0 or -1.0)
/// None if forbidden excitation
pub fn apply_excitation(string: u64, p: usize, q: usize) -> Option<(u64, f64)> {
    let bit_q = 1u64 << q;
    let bit_p = 1u64 << p;

    // is q occupied?
    if (string & bit_q) == 0 {
        return None;
    }

    // is p already occupied? (instead of opposite spin)
    if p != q && (string & bit_p) != 0 {
        return None;
    }

    // if p == q, it's not really an excitation but allow for convenience
    if p == q {
        return Some((string, 1.0));
    }

    // fermionic phase shift (count how many occupied orbitals are between p and q)
    let min_idx = p.min(q);
    let max_idx = p.max(q);
    
    // bitmasks between p and q
    let length = max_idx - min_idx - 1;
    let phase = if length == 0 {
        1.0 // No orbitals between p and q -> Phase remains positive
    } else {
        let mask = ((1u64 << length) - 1) << (min_idx + 1);
        let electrons_between = (string & mask).count_ones(); // count number of ones in binary representation
        
        // Phase changes sign for every occupied orbital between p and q
        if electrons_between % 2 == 1 { -1.0 } else { 1.0 }
    };

    // anhilation and creation via XOR 
    let new_string = string ^ bit_p ^ bit_q;

    Some((new_string, phase))
}

fn get_ao_integral(mu: usize, nu: usize, lambda: usize, sigma: usize, packed_2e: &[f64]) -> f64 {
    let ij = get_integral_index(mu, nu);
    let kl = get_integral_index(lambda, sigma);
    let idx = get_integral_index(ij, kl);
    packed_2e[idx]
}

pub fn transform_integrals_ao_to_mo(
    ao_integrals: &[f64], 
    c_matrix: &[f64], // c[mo_index * n_basis + ao_index]
    n_basis: usize
) -> Vec<f64> {
    
    // flat indices for 4D array: (p, q, r, s) -> p * n_basis^3 + q * n_basis^2 + r * n_basis + s
    let idx = |p: usize, q: usize, r: usize, s: usize| {
        ((p * n_basis + q) * n_basis + r) * n_basis + s
    };

    let mut temp1 = vec![0.0; n_basis.pow(4)];
    let mut temp2 = vec![0.0; n_basis.pow(4)];
    let mut temp3 = vec![0.0; n_basis.pow(4)];
    let mut mo_ints = vec![0.0; n_basis.pow(4)];


    print!("Step 1/4: Quarter-Transformation... ");
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for lam in 0..n_basis {
                for s in 0..n_basis {
                    let mut sum = 0.0;
                    for sig in 0..n_basis {
                        //  c[MO * n_basis + AO]
                        let c_sig_s = c_matrix[s * n_basis + sig];
                        // (mu, nu | lam, sig)
                        sum += c_sig_s * get_ao_integral(mu, nu, lam, sig, ao_integrals); 
                    }
                    temp1[idx(mu, nu, lam, s)] = sum;
                }
            }
        }
    }
    println!("Done.");


    print!("Step 2/4: Half-Transformation...    ");
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for r in 0..n_basis {
                for s in 0..n_basis {
                    let mut sum = 0.0;
                    for lam in 0..n_basis {
                        let c_lam_r = c_matrix[r * n_basis + lam];
                        sum += c_lam_r * temp1[idx(mu, nu, lam, s)];
                    }
                    temp2[idx(mu, nu, r, s)] = sum;
                }
            }
        }
    }
    println!("Done.");


    print!("Step 3/4: Three-Quarter-Transform...");
    for mu in 0..n_basis {
        for q in 0..n_basis {
            for r in 0..n_basis {
                for s in 0..n_basis {
                    let mut sum = 0.0;
                    for nu in 0..n_basis {
                        let c_nu_q = c_matrix[q * n_basis + nu];
                        sum += c_nu_q * temp2[idx(mu, nu, r, s)];
                    }
                    temp3[idx(mu, q, r, s)] = sum;
                }
            }
        }
    }
    println!("Done.");


    print!("Step 4/4: Full MO-Transformation... ");
    for p in 0..n_basis {
        for q in 0..n_basis {
            for r in 0..n_basis {
                for s in 0..n_basis {
                    let mut sum = 0.0;
                    for mu in 0..n_basis {
                        let c_mu_p = c_matrix[p * n_basis + mu];
                        sum += c_mu_p * temp3[idx(mu, q, r, s)];
                    }
                    mo_ints[idx(p, q, r, s)] = sum;
                }
            }
        }
    }
    println!("Done.");

    mo_ints
}

pub fn compute_sigma_vector(
    dets: &[Determinant],
    c_vector: &[f64],
    h_mo: &[f64],
    eri_mo: &[f64],
    h_diag: &[f64], 
    n_basis: usize,
) -> Vec<f64> {
    
    // lookup table
    let det_index: HashMap<_, _> = dets.iter().enumerate().map(|(i, &d)| (d, i)).collect();

    // helper function to access MO integrals in 4D format from flat array
    let get_eri = |p: usize, q: usize, r: usize, s: usize| {
        eri_mo[p * n_basis.pow(3) + q * n_basis.pow(2) + r * n_basis + s]
    };

    // 
    let sigma = dets.par_iter().enumerate().fold(
        || vec![0.0; dets.len()], // each thread gets its own local sigma vector
        |mut local_sigma, (i, &det)| {
            let ci = c_vector[i];
            if ci.abs() < 1e-12 { return local_sigma; }

            // ==========================================
            // 0 e
            // ==========================================
            local_sigma[i] += h_diag[i] * ci;

            // ==========================================
            // 1 e
            // ==========================================
            for r in 0..n_basis {
                for p in 0..n_basis {
                    if p == r { continue; } // Keine Diagonale!

                    // --- Alpha  ---
                    if let Some((new_alpha, phase)) = apply_excitation(det.alpha, p, r) {
                        let new_det = Determinant { alpha: new_alpha, beta: det.beta };
                        if let Some(&j) = det_index.get(&new_det) {
                            let mut h_ij = h_mo[p * n_basis + r];
                            for k in 0..n_basis {
                                if (det.alpha & (1 << k)) != 0 { h_ij += get_eri(p, r, k, k) - get_eri(p, k, k, r); }
                                if (det.beta  & (1 << k)) != 0 { h_ij += get_eri(p, r, k, k); }
                            }
                            local_sigma[j] += phase * h_ij * ci;
                        }
                    }

                    // --- Beta ---
                    if let Some((new_beta, phase)) = apply_excitation(det.beta, p, r) {
                        let new_det = Determinant { alpha: det.alpha, beta: new_beta };
                        if let Some(&j) = det_index.get(&new_det) {
                            let mut h_ij = h_mo[p * n_basis + r];
                            for k in 0..n_basis {
                                if (det.beta  & (1 << k)) != 0 { h_ij += get_eri(p, r, k, k) - get_eri(p, k, k, r); }
                                if (det.alpha & (1 << k)) != 0 { h_ij += get_eri(p, r, k, k); }
                            }
                            local_sigma[j] += phase * h_ij * ci;
                        }
                    }
                }
            }

            // ==========================================
            // 2 e
            // ==========================================
            
            // --- Alpha-Beta ---
            for r in 0..n_basis {
                for p in 0..n_basis {
                    if p == r { continue; }
                    if let Some((new_alpha, phase_a)) = apply_excitation(det.alpha, p, r) {
                        for s in 0..n_basis {
                            for q in 0..n_basis {
                                if q == s { continue; }
                                if let Some((new_beta, phase_b)) = apply_excitation(det.beta, q, s) {
                                    let new_det = Determinant { alpha: new_alpha, beta: new_beta };
                                    if let Some(&j) = det_index.get(&new_det) {
                                        local_sigma[j] += phase_a * phase_b * get_eri(p, r, q, s) * ci;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // --- Alpha-Alpha ---
            for r in 0..n_basis {
                for s in (r + 1)..n_basis {
                    if (det.alpha & (1 << r)) == 0 || (det.alpha & (1 << s)) == 0 { continue; }
                    for p in 0..n_basis {
                        for q in (p + 1)..n_basis {
                            if p == r || p == s || q == r || q == s { continue; }

                            let temp_alpha = det.alpha & !(1 << r) & !(1 << s);
                            if (temp_alpha & (1 << p)) != 0 || (temp_alpha & (1 << q)) != 0 { continue; }
                            
                            let new_alpha = temp_alpha | (1 << p) | (1 << q);
                            let new_det = Determinant { alpha: new_alpha, beta: det.beta };
                            
                            if let Some(&j) = det_index.get(&new_det) {
                                let mut phase = 1.0;
                                let mut t = det.alpha;
                                phase *= get_phase(t, r); t &= !(1 << r);
                                phase *= get_phase(t, s); t &= !(1 << s);
                                phase *= get_phase(t, q); t |= (1 << q);
                                phase *= get_phase(t, p);
                                
                                let h_ij = get_eri(p, r, q, s) - get_eri(p, s, q, r);
                                local_sigma[j] += phase * h_ij * ci;
                            }
                        }
                    }
                }
            }

            // --- Beta-Beta ---
            for r in 0..n_basis {
                for s in (r + 1)..n_basis {
                    if (det.beta & (1 << r)) == 0 || (det.beta & (1 << s)) == 0 { continue; }
                    for p in 0..n_basis {
                        for q in (p + 1)..n_basis {
                            if p == r || p == s || q == r || q == s { continue; }
                            
                            let temp_beta = det.beta & !(1 << r) & !(1 << s);
                            if (temp_beta & (1 << p)) != 0 || (temp_beta & (1 << q)) != 0 { continue; }
                            
                            let new_beta = temp_beta | (1 << p) | (1 << q);
                            let new_det = Determinant { alpha: det.alpha, beta: new_beta };
                            
                            if let Some(&j) = det_index.get(&new_det) {
                                let mut phase = 1.0;
                                let mut t = det.beta;
                                phase *= get_phase(t, r); t &= !(1 << r);
                                phase *= get_phase(t, s); t &= !(1 << s);
                                
                                phase *= get_phase(t, q); t |= (1 << q);
                                phase *= get_phase(t, p);
                                
                                let h_ij = get_eri(p, r, q, s) - get_eri(p, s, q, r);
                                local_sigma[j] += phase * h_ij * ci;
                            }
                        }
                    }
                }
            }

            local_sigma 
        }
    ).reduce(
        || vec![0.0; dets.len()], 
        |mut combined, local| {

            for (c, l) in combined.iter_mut().zip(local.iter()) {
                *c += l;
            }
            combined
        }
    );

    sigma
}

pub fn compute_diagonal(
    dets: &[Determinant],
    h_mo: &[f64],
    eri_mo: &[f64],
    n_basis: usize,
) -> Vec<f64> {
    let mut diag = Vec::with_capacity(dets.len());

    let get_eri = |p: usize, q: usize, r: usize, s: usize| {
        eri_mo[p * n_basis.pow(3) + q * n_basis.pow(2) + r * n_basis + s]
    };

    for det in dets {
        let mut e = 0.0;

        // --- 1-ELECTRON-PART ---
        for p in 0..n_basis {
            if (det.alpha & (1 << p)) != 0 { e += h_mo[p * n_basis + p]; }
            if (det.beta & (1 << p)) != 0 { e += h_mo[p * n_basis + p]; }
        }

        // --- 2-ELECTRON-PART ---
        for p in 0..n_basis {
            for q in 0..n_basis {
                let in_alpha_p = (det.alpha & (1 << p)) != 0;
                let in_alpha_q = (det.alpha & (1 << q)) != 0;
                let in_beta_p = (det.beta & (1 << p)) != 0;
                let in_beta_q = (det.beta & (1 << q)) != 0;

                // Alpha-Alpha (Coulomb - Exchange)
                if in_alpha_p && in_alpha_q && p < q {
                    e += get_eri(p, p, q, q) - get_eri(p, q, p, q);
                }
                // Beta-Beta (Coulomb - Exchange)
                if in_beta_p && in_beta_q && p < q {
                    e += get_eri(p, p, q, q) - get_eri(p, q, p, q);
                }
                // Alpha-Beta (Only Coulomb)
                if in_alpha_p && in_beta_q {
                    e += get_eri(p, p, q, q);
                }
            }
        }
        diag.push(e);
    }
    diag
}

use lapack::dspev;

pub fn diagonalize_subspace(packed_matrix: &mut [f64], dim: usize) -> (f64, Vec<f64>) {
    let mut eigval = vec![0.0; dim];
    let mut eigvec = vec![0.0; dim * dim];
    let mut work = vec![0.0; 3 * dim];
    let mut info = 0;

    unsafe {
        dspev(b'V', b'U', dim as i32, packed_matrix, &mut eigval, &mut eigvec, dim as i32, &mut work, &mut info);
    }
    if info != 0 { panic!("Subspace diagonalization failed!"); }

    let lowest_eval = eigval[0];
    let mut lowest_evec = vec![0.0; dim];
    for i in 0..dim {
        lowest_evec[i] = eigvec[i]; // Erste Spalte von eigvec
    }

    (lowest_eval, lowest_evec)
}

pub fn transform_1e_ao_to_mo(h_ao: &[f64], c_matrix: &[f64], n_basis: usize) -> Vec<f64> {
    let mut h_mo = vec![0.0; n_basis * n_basis];
    let mut temp = vec![0.0; n_basis * n_basis];
    
    // temp = H_AO * C
    for mu in 0..n_basis {
        for p in 0..n_basis {
            let mut sum = 0.0;
            for nu in 0..n_basis {
                sum += h_ao[mu * n_basis + nu] * c_matrix[p * n_basis + nu];
            }
            temp[mu * n_basis + p] = sum;
        }
    }
    
    // H_MO = C^T * temp
    for q in 0..n_basis {
        for p in 0..n_basis {
            let mut sum = 0.0;
            for mu in 0..n_basis {
                // c_matrix ist C[MO * n_basis + AO], daher q * n_basis + mu
                sum += c_matrix[q * n_basis + mu] * temp[mu * n_basis + p];
            }
            h_mo[q * n_basis + p] = sum;
        }
    }
    h_mo
}

pub fn do_davidson(
    dets: &[Determinant],
    h_mo: &[f64],
    eri_mo: &[f64],
    n_basis: usize,
    max_iter: usize,
    tol: f64
) -> (f64, Vec<f64>) {
    
    let n_dets = dets.len();
    let h_diag = compute_diagonal(dets, h_mo, eri_mo, n_basis);

    // initial guess hf determinant
    let mut b0 = vec![0.0; n_dets];
    b0[0] = 1.0; 
    
    // Arrays for basis vectors and sigma vectors
    let mut b_vecs = vec![b0];
    let mut s_vecs = Vec::new();

    let mut current_energy = 0.0;
    let mut current_c_vector = vec![0.0; n_dets];

    println!("\n=========== DAVIDSON DIAGONALIZATION ===========");
    println!("Iter |    FCI Energy (a.u.)   | Residual Norm");
    println!("------------------------------------------------");

    for iter in 0..max_iter {
        let dim = b_vecs.len();

        // Multiply basis vector with the H-Matrix
        let new_sigma = compute_sigma_vector(dets, &b_vecs[dim - 1], h_mo, eri_mo, &h_diag, n_basis);
        s_vecs.push(new_sigma);

        // 3. Subspace-Matrix H_sub = B^T * H * B 
        let mut h_sub = Vec::with_capacity((dim * (dim + 1)) / 2);
        for j in 0..dim {
            for i in 0..=j {
                let element: f64 = b_vecs[i].iter().zip(s_vecs[j].iter()).map(|(b, s)| b * s).sum();
                h_sub.push(element);
            }
        }

        // diagonalize subspace matrix to get lowest eigenvalue and corresponding eigenvector
        let (eval, evec_sub) = diagonalize_subspace(&mut h_sub, dim);
        current_energy = eval;

        // calculate Ritz vector in full space: c = B * evec_sub
        current_c_vector.fill(0.0);
        let mut ritz_sigma = vec![0.0; n_dets];
        for k in 0..dim {
            let alpha = evec_sub[k];
            for i in 0..n_dets {
                current_c_vector[i] += alpha * b_vecs[k][i];
                ritz_sigma[i]       += alpha * s_vecs[k][i];
            }
        }

        // r = H*c - E*c
        let mut residual = vec![0.0; n_dets];
        let mut res_norm_sq = 0.0;
        for i in 0..n_dets {
            residual[i] = ritz_sigma[i] - current_energy * current_c_vector[i];
            res_norm_sq += residual[i].powi(2);
        }
        let res_norm = res_norm_sq.sqrt();

        println!(" {:>3} | {:>22.12} | {:>10.3e}", iter + 1, current_energy, res_norm);

        // convergence check
        if res_norm < tol {
            println!("================================================");
            println!("Davidson converged successfully!");
            break;
        }

        // weighting the residual by the inverse of the diagonal (preconditioning)
        let mut new_b = vec![0.0; n_dets];
        for i in 0..n_dets {
            let denominator = current_energy - h_diag[i];
  
            if denominator.abs() > 1e-6 {
                new_b[i] = residual[i] / denominator;
            } else {
                new_b[i] = residual[i];
            }
        }

        // gram schmidt orthogonalisierung gegen alle bisherigen basisvektoren
        for k in 0..dim {
            let overlap: f64 = new_b.iter().zip(b_vecs[k].iter()).map(|(nb, b)| nb * b).sum();
            for i in 0..n_dets {
                new_b[i] -= overlap * b_vecs[k][i];
            }
        }

        // normalize
        let norm_sq: f64 = new_b.iter().map(|x| x.powi(2)).sum();
        if norm_sq > 1e-12 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for val in new_b.iter_mut() { *val *= inv_norm; }
            b_vecs.push(new_b);
        } else {
            // if vecot is linear dependent on basis, skip
            break; 
        }
    }

    (current_energy, current_c_vector)
}

pub fn analyze_fci_vector(
    dets: &[Determinant], 
    c_vector: &[f64], 
    n_basis: usize, 
    threshold: f64
) {
    println!("\n==================================================================");
    println!(" F C I   W A V E F U N C T I O N   A N A L Y S I S");
    println!("==================================================================");
    println!("   Weight | Coefficient | Alpha String | Beta String");
    println!("------------------------------------------------------------------");

    let mut indices: Vec<usize> = (0..dets.len()).collect();
    indices.sort_by(|&a, &b| c_vector[b].abs().partial_cmp(&c_vector[a].abs()).unwrap());

    let mut total_weight = 0.0;

    for &i in &indices {
        let c = c_vector[i];
        let weight = c * c;
        total_weight += weight;

        if weight > threshold {
            let det = dets[i];
            // rust bitstring magic
            println!("{:8.4}% | {:>11.6} | {:0width$b} | {:0width$b}", 
                weight * 100.0, 
                c, 
                det.alpha, 
                det.beta, 
                width = n_basis
            );
        }
    }
    println!("------------------------------------------------------------------");
    println!("Total Weight represented above: {:.4}%", total_weight * 100.0);
    println!("==================================================================\n");
}

pub fn read_fcidump(filename: &str) -> (usize, usize, f64, Vec<f64>, Vec<f64>) {
    let file = File::open(filename).expect("FCIDUMP Datei konnte nicht geöffnet werden!");
    let reader = BufReader::new(file);
    
    let mut n_basis = 0;
    let mut n_elec = 0;
    let mut e_core = 0.0;
    
    let mut h_mo = Vec::new();
    let mut eri_mo = Vec::new();
    let mut header_done = false;

    for line_res in reader.lines() {
        let line = line_res.unwrap();
        let line = line.trim();
        if line.is_empty() { continue; }

        if !header_done {
            if line.contains("NORB") {
                let parts: Vec<&str> = line.split(&[',', '=']).collect();
                for (idx, p) in parts.iter().enumerate() {
                    if p.trim() == "NORB" || p.trim() == "&FCI NORB" {
                        n_basis = parts[idx + 1].trim().parse().unwrap();
                    }
                    if p.trim() == "NELEC" {
                        n_elec = parts[idx + 1].trim().parse().unwrap();
                    }
                }
            }
            if line.contains("&END") || line == "/" {
                header_done = true;
                h_mo = vec![0.0; n_basis * n_basis];
                eri_mo = vec![0.0; n_basis.pow(4)];
            }
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 5 { continue; }
        
        let val: f64 = tokens[0].parse().unwrap();
        let i: usize = tokens[1].parse().unwrap();
        let j: usize = tokens[2].parse().unwrap();
        let k: usize = tokens[3].parse().unwrap();
        let l: usize = tokens[4].parse().unwrap();

        if i == 0 && j == 0 && k == 0 && l == 0 {
            e_core = val;
        } else if k == 0 && l == 0 {
            let p = i - 1; let q = j - 1;
            h_mo[p * n_basis + q] = val;
            h_mo[q * n_basis + p] = val; 
        } else {
            let p = i - 1; let q = j - 1; let r = k - 1; let s = l - 1;
            let mut set_eri = |a: usize, b: usize, c: usize, d: usize| {
                eri_mo[a * n_basis.pow(3) + b * n_basis.pow(2) + c * n_basis + d] = val;
            };
            set_eri(p, q, r, s); set_eri(q, p, r, s);
            set_eri(p, q, s, r); set_eri(q, p, s, r);
            set_eri(r, s, p, q); set_eri(s, r, p, q);
            set_eri(r, s, q, p); set_eri(s, r, q, p);
        }
    }
    (n_basis, n_elec, e_core, h_mo, eri_mo)
}