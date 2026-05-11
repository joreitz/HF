use rayon::prelude::*;
use crate::types::StaticHFData;
use crate::parser::OptFlags;
use std::io::{self, Write};

pub struct CenterTask {
    pub mu: usize,
    pub axis: usize, 
    pub step: f64
}

pub struct GaussianTask {
    pub param_idx: usize,
    pub is_alpha: bool, 
    pub step: f64,
}

pub struct NuclearTask {
    pub atom_idx: usize,
    pub axis: usize, 
    pub step: f64,
}

pub fn calculate_nuclear_gradients(
    base_xyz: &[[f64; 3]],
    static_data: &StaticHFData,
    p_matrix_ref: &[f64],
) -> Vec<[f64; 3]> {
    let step = 0.001; 
    let n_atoms = base_xyz.len();
    let mut tasks = Vec::new();

    for atom_idx in 0..n_atoms {
        if static_data.z_nuc[atom_idx] > 0.0 {
            for axis in 0..3 {
                tasks.push(NuclearTask { atom_idx, axis, step });
                tasks.push(NuclearTask { atom_idx, axis, step: -step });
            }
        }
    }

    // energy calculations in parallel
    let energies: Vec<f64> = tasks.par_iter().map(|task| {
        let mut local_xyz = base_xyz.to_vec();
        let mut local_data = static_data.clone();
        
        // shift nuclei
        local_xyz[task.atom_idx][task.axis] += task.step;
        
        // shift basis functions, that stay on nuclei
        for mu in 0..local_data.n_basis {
            if local_data.basis_to_atom[mu] == task.atom_idx {
                local_data.centers[mu][task.axis] += task.step;
            }
        }
        
        let scf_result = crate::driver::calculate_single_point_energy(
            &local_xyz, &local_data, Some(p_matrix_ref), true
        );
        
        scf_result.energy
    }).collect();


    let mut gradients = vec![[0.0, 0.0, 0.0]; n_atoms];
    let mut task_idx = 0;

    for atom_idx in 0..n_atoms {
        if static_data.z_nuc[atom_idx] > 0.0 {
            for axis in 0..3 {
                let e_plus = energies[task_idx];
                let e_minus = energies[task_idx + 1];
                gradients[atom_idx][axis] = (e_plus - e_minus) / (2.0 * step);
                task_idx += 2;
            }
        }
    }

    gradients
}

pub fn do_optimization(
    base_xyz: &[[f64; 3]],
    static_data: &crate::types::StaticHFData,
    mut p_matrix_current: Vec<f64>, 
    max_steps: usize,
    convergence_threshold: f64,
) -> Vec<[f64; 3]> {
    
    let mut current_xyz = base_xyz.to_vec();
    let mut velocity = vec![[0.0; 3]; current_xyz.len()];
    let learning = 0.3;
    let momentum = 0.6;
    let max_step = 0.2;

    println!("\n=====================================================================");
    println!(" G E O M E T R Y   O P T I M I Z A T I O N   ");
    println!("=====================================================================");
    println!(" Step |   Energy (a.u.)  |  Δ Energy (a.u.) |   Max Grad  |  Grad Norm");
    println!("---------------------------------------------------------------------");

    let mut last_energy = 0.0;

    for step in 0..max_steps {
        let scf_result = crate::driver::calculate_single_point_energy(&current_xyz, static_data, Some(&p_matrix_current), true);
        p_matrix_current = scf_result.p_matrix;

        let gradient = calculate_nuclear_gradients(&current_xyz, static_data, &p_matrix_current);

        let delta_e = if step == 0 { 0.0 } else { scf_result.energy - last_energy };
        last_energy = scf_result.energy;

        let grad_norm: f64 = gradient.iter().flat_map(|g| g.iter()).map(|&x| x.powi(2)).sum::<f64>().sqrt();
        let max_grad = gradient.iter().flat_map(|g| g.iter()).map(|&x| x.abs()).fold(0.0_f64, f64::max);
        
        println!(" {:>4} | {:>16.8} | {:>16.8} | {:>11.6} | {:>10.6}", 
            step, scf_result.energy, delta_e, max_grad, grad_norm);
        
        if max_grad < convergence_threshold {
            println!("---------------------------------------------------------------------");
            println!(" *** Optimization converged! (Max Gradient < {}) ***", convergence_threshold);
            break;
        }
        
        for i in 0..current_xyz.len() {
            for axis in 0..3 {
                velocity[i][axis] = momentum * velocity[i][axis] - learning * gradient[i][axis];
                
                let mut step_size = velocity[i][axis];
                if step_size > max_step { step_size = max_step; }
                if step_size < -max_step { step_size = -max_step; }

                current_xyz[i][axis] += step_size;
            }
        }

        if step == max_steps - 1 {
            println!("---------------------------------------------------------------------");
            println!(" WARNING: Optimization did NOT converge in {} steps.", max_steps);
        }

    }
    current_xyz
}

pub fn calculate_gaussian_gradients(
    base_xyz: &[[f64; 3]],
    static_data: &StaticHFData,
    p_matrix_ref: &[f64],
) -> (Vec<f64>, Vec<f64>) { 
    
    let step = 0.001; 
    let total_primitives = static_data.n_basis * static_data.ng;
    
    // 4 Tasks per Primitive (Alpha+, Alpha-, Coeff+, Coeff-)
    let mut tasks = Vec::with_capacity(total_primitives * 4);

    for i in 0..total_primitives {
        // Alphas
        tasks.push(GaussianTask { param_idx: i, is_alpha: true, step: step });
        tasks.push(GaussianTask { param_idx: i, is_alpha: true, step: -step });
        // Coeffs
        tasks.push(GaussianTask { param_idx: i, is_alpha: false, step: step });
        tasks.push(GaussianTask { param_idx: i, is_alpha: false, step: -step });
    }

    // .par_iter() for parallelization
    let energies: Vec<f64> = tasks.par_iter().map(|task| {
        let mut local_data = static_data.clone();
        
        if task.is_alpha {
            local_data.alphas[task.param_idx] += task.step;
            
            // 
            if local_data.alphas[task.param_idx] < 0.01 {
                local_data.alphas[task.param_idx] = 0.01;
            }
        } else {
            local_data.coeffs[task.param_idx] += task.step;
        }
        
        let scf_result = crate::driver::calculate_single_point_energy(
            base_xyz, &local_data, Some(p_matrix_ref), true
        );
        
        scf_result.energy
    }).collect();

    // Gradient
    let mut grad_alphas = vec![0.0; total_primitives];
    let mut grad_coeffs = vec![0.0; total_primitives];
    let mut task_idx = 0;

    for i in 0..total_primitives {
        // Alphas
        let e_alpha_plus = energies[task_idx];
        let e_alpha_minus = energies[task_idx + 1];
        grad_alphas[i] = (e_alpha_plus - e_alpha_minus) / (2.0 * step); 
        task_idx += 2;

        // Coeffs
        let e_coeff_plus = energies[task_idx];
        let e_coeff_minus = energies[task_idx + 1];
        grad_coeffs[i] = (e_coeff_plus - e_coeff_minus) / (2.0 * step); 
        task_idx += 2;
    }

    (grad_alphas, grad_coeffs)
}

pub fn calculate_center_gradients(
    base_xyz: &[[f64; 3]],
    static_data: &StaticHFData,
    p_matrix_ref: &[f64],
) -> Vec<[f64; 3]> {
    
    let step = 0.001; 
    let n_basis = static_data.n_basis;
    let mut tasks = Vec::new();

    for mu in 0..n_basis {
        let atom_idx = static_data.basis_to_atom[mu];
        if static_data.z_nuc[atom_idx] == 0.0 {
            for axis in 0..3 {
                tasks.push(CenterTask { mu, axis, step });
                tasks.push(CenterTask { mu, axis, step: -step });
            }
        }
    }

    let energies: Vec<f64> = tasks.par_iter().map(|task| {
        let mut local_data = static_data.clone();
        
        local_data.centers[task.mu][task.axis] += task.step;
        
        let scf_result = crate::driver::calculate_single_point_energy(
            base_xyz, &local_data, Some(p_matrix_ref), true
        );
        
        scf_result.energy
    }).collect();

    let mut gradients = vec![[0.0, 0.0, 0.0]; n_basis];
    let mut task_idx = 0;

    for mu in 0..n_basis {
        let atom_idx = static_data.basis_to_atom[mu];
        if static_data.z_nuc[atom_idx] == 0.0 {
            for axis in 0..3 {
                let e_plus = energies[task_idx];
                let e_minus = energies[task_idx + 1];
                gradients[mu][axis] = (e_plus - e_minus) / (2.0 * step);
                task_idx += 2;
            }
        }
    }

    gradients
}

pub fn do_master_optimization(
    base_xyz: &[[f64; 3]],
    static_data: &StaticHFData,
    p_matrix_ref: &[f64],
    max_steps: usize,
    convergence_threshold: f64,
    flags: OptFlags,
) -> (Vec<[f64; 3]>, StaticHFData) { 
    
    let mut current_xyz = base_xyz.to_vec();
    let mut current_static_data = static_data.clone();
    let mut p_matrix_current = p_matrix_ref.to_vec();

    println!("\n=====================================================================");
    println!(" O P T I M I Z A T I O N");
    println!(" Active Modules: {:?}", flags);
    println!("=====================================================================");
    println!(" Step |   Energy (a.u.)  |  Δ Energy (a.u.) |   Max Grad  |  Grad Norm");
    println!("---------------------------------------------------------------------");

    // Adam Hyperparameter
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let epsilon = 1e-8_f64;

    let base_lr_basis = 0.05_f64;  
    let base_lr_geom  = 0.002_f64; 

    let n_basis = current_static_data.n_basis;
    let total_primitives = n_basis * current_static_data.ng;
    let n_atoms = current_xyz.len();

    // Moment-Speicher (m = 1st moment, v = 2nd moment)
    let mut m_geom = vec![[0.0; 3]; n_atoms];
    let mut v_geom = vec![[0.0; 3]; n_atoms];
    
    let mut m_centers = vec![[0.0; 3]; n_basis];
    let mut v_centers = vec![[0.0; 3]; n_basis];
    
    let mut m_alpha = vec![0.0; total_primitives];
    let mut v_alpha = vec![0.0; total_primitives];
    
    let mut m_coeff = vec![0.0; total_primitives];
    let mut v_coeff = vec![0.0; total_primitives];

    let mut last_energy = 0.0;
    
    for step in 0..max_steps {
        let scf_result = crate::driver::calculate_single_point_energy(
            base_xyz, &current_static_data, Some(&p_matrix_current), true
        );
        p_matrix_current = scf_result.p_matrix;
        
        let current_energy = scf_result.energy;
        let delta_e = if step == 0 { 0.0 } else { current_energy - last_energy };
        last_energy = current_energy;

        let mut global_max_grad = 0.0_f64;
        let mut global_grad_norm_sq = 0.0_f64;

        let mut geom_grads = None;
        // NUC OPT
        if flags.contains(OptFlags::GEOMETRY) {
            let grads = calculate_nuclear_gradients(&current_xyz, &current_static_data, &p_matrix_current);
            for a in 0..current_xyz.len() {
                for axis in 0..3 {
                    let g = grads[a][axis];
                    global_max_grad = global_max_grad.max(g.abs());
                    global_grad_norm_sq += g.powi(2);
                }
            }
            geom_grads = Some(grads);
        }

        // FSGO OPT
        let mut center_grads = None;
        if flags.contains(OptFlags::FSGO_CENTERS) {
            let grads = calculate_center_gradients(base_xyz, &current_static_data, &p_matrix_current);
            for mu in 0..current_static_data.n_basis {
                for axis in 0..3 {
                    let g = grads[mu][axis];
                    global_max_grad = global_max_grad.max(g.abs());
                    global_grad_norm_sq += g.powi(2);
                }
            }
            center_grads = Some(grads);
        }

        // GTO BASIS (ALPHAS/COEFFS) OPT 
        let mut gaussian_grads = None;
        if flags.contains(OptFlags::GAUSSIANS) {
            let (grad_alphas, grad_coeffs) = calculate_gaussian_gradients(base_xyz, &current_static_data, &p_matrix_current);
            let total_primitives = current_static_data.n_basis * current_static_data.ng;
            for i in 0..total_primitives {
                global_max_grad = global_max_grad.max(grad_alphas[i].abs()).max(grad_coeffs[i].abs());
                global_grad_norm_sq += grad_alphas[i].powi(2) + grad_coeffs[i].powi(2);
            }
            gaussian_grads = Some((grad_alphas, grad_coeffs));
        }

        let global_grad_norm = global_grad_norm_sq.sqrt();
        
        // --- LIVE-UPDATE IM TERMINAL ---
        print!("\r {:>4} | {:>16.8} | {:>16.8} | {:>11.6} | {:>10.6}", 
            step, current_energy, delta_e, global_max_grad, global_grad_norm);
        
        // Zwingt das Terminal, die Zeile SOFORT anzuzeigen (wichtig bei print!)
        io::stdout().flush().unwrap();
        
        // KONVERGENZ-CHECK
        if global_max_grad < convergence_threshold && step > 0 {
            println!(); // Macht einen Zeilenumbruch, damit die letzte Zeile stehen bleibt!
            println!("Master Optimization converged! (Max Gradient < {})", convergence_threshold);
            break;
        }
        
        // convergence
        if global_max_grad < convergence_threshold && step > 0 {
            println!(); 
            println!("Master Optimization converged! (Max Gradient < {})", convergence_threshold);
            break;
        }
        
        // 2. Kriterium: Das "Noise-Floor" 
        if step > 5 && delta_e.abs() < 1e-6 {
            println!();
            println!("Master Optimization converged! Energy change hit the numerical noise floor (ΔE < 1e-6).");
            break;
        }

        // updating parameters
        let t = (step + 1) as f64; // Adam Timestep (beginnt bei 1)

        let current_lr_basis = base_lr_basis / (1.0 + 0.01 * step as f64);
        let current_lr_geom  = base_lr_geom  / (1.0 + 0.01 * step as f64);

        // Adam-converger
        let apply_adam = |grad: f64, m: &mut f64, v: &mut f64, lr: f64| -> f64 {
            *m = beta1 * (*m) + (1.0 - beta1) * grad;
            *v = beta2 * (*v) + (1.0 - beta2) * grad.powi(2);
            let m_hat = *m / (1.0 - beta1.powi(t as i32));
            let v_hat = *v / (1.0 - beta2.powi(t as i32));
            lr * m_hat / (v_hat.sqrt() + epsilon) 
        };

        if let Some(grads) = geom_grads {
            let sd_lr_geom = 0.05_f64; // 
            for a in 0..n_atoms {
                if current_static_data.z_nuc[a] > 0.0 {
                    for axis in 0..3 {
                        let step_val = sd_lr_geom * grads[a][axis];
                        current_xyz[a][axis] -= step_val;
                        
                        for mu in 0..n_basis {
                            if current_static_data.basis_to_atom[mu] == a {
                                current_static_data.centers[mu][axis] -= step_val;
                            }
                        }
                    }
                }
            }
        }

        if let Some(grads) = center_grads {
            let sd_lr_centers = 0.1_f64; 
            for mu in 0..n_basis {
                for axis in 0..3 {
                    let step_val = sd_lr_centers * grads[mu][axis];
                    current_static_data.centers[mu][axis] -= step_val;
                }
            }
        }

        if let Some((grad_alphas, grad_coeffs)) = gaussian_grads {
            for i in 0..total_primitives {
                // Alphas update
                let step_alpha = apply_adam(grad_alphas[i], &mut m_alpha[i], &mut v_alpha[i], current_lr_basis);
                current_static_data.alphas[i] -= step_alpha;
                if current_static_data.alphas[i] < 0.05 { current_static_data.alphas[i] = 0.05; } // Safety
                
                // Coeffs update
                let step_coeff = apply_adam(grad_coeffs[i], &mut m_coeff[i], &mut v_coeff[i], current_lr_basis);
                current_static_data.coeffs[i] -= step_coeff;
            }
        }
    }
    
    let output_filename = "optimized_master.out";
    match crate::io_helper::export_to_input_format(output_filename, &current_static_data) {
        Ok(_) => println!("Optimized structe has been saved to '{}'.", output_filename),
        Err(e) => eprintln!("Error with writing to {}: {}", output_filename, e),
    }
    
    // Geometrie exportieren
    let output_filename = "optimized_master.out";
    match crate::io_helper::export_to_input_format(output_filename, &current_static_data) {
        Ok(_) => {
            println!("\n=====================================================================");
            println!(" OPTIMIZATION COMPLETE");
            println!("=====================================================================");
            println!(" Final Energy:      {:.12} Hartree", last_energy);
            println!(" Output File:       {}", output_filename);
            println!("=====================================================================\n");
        },
        Err(e) => eprintln!("Fehler beim Schreiben der Out-Datei: {}", e),
    }

    (current_xyz, current_static_data)
}