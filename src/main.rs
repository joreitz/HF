use std::fs;
use std::env;
use std::process;
use lapack::dspev;
extern crate intel_mkl_src;

mod mp2;
mod parser;
mod slater;
mod integrals;
mod population;
mod scf;
mod types;
mod driver; 
mod grad;
mod print_matrix;
mod io_helper;
mod fci;

use crate::parser::{parse_hf_input}; 
use population::mulliken_pop_analysis;
use crate::types::StaticHFData;
use crate::driver::calculate_single_point_energy;
use crate::fci::{do_davidson, build_fci_space, transform_1e_ao_to_mo, transform_integrals_ao_to_mo, read_fcidump};

use crate::grad::{do_optimization, do_master_optimization};

pub fn nuc_coulomb_repulsion(r_a: &[f64; 3], r_b: &[f64; 3], z_a: f64, z_b: f64) -> f64 {
    let rab: f64 = r_a.iter().zip(r_b.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    z_a * z_b / rab.sqrt()
}

pub fn calc_nuc_rep(xyz: &[[f64; 3]], chrg: &[f64]) -> f64 {
    let mut enuc = 0.0;
    let n_atoms = xyz.len();
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            if chrg[i] == 0.0 || chrg[j] == 0.0 {
                continue;
            }
            enuc += nuc_coulomb_repulsion(&xyz[i], &xyz[j], chrg[i], chrg[j]);
        }
    }
    enuc
}

pub fn sym_matrix_packing(matrix: &[f64], dim: usize) -> Vec<f64> {
    let mut packed = Vec::with_capacity(dim * (dim + 1) / 2);
    for j in 0..dim {
        for i in 0..=j {
            packed.push(0.5 * (matrix[j * dim + i] + matrix[i * dim + j]));
        }
    }
    packed
}

pub fn unpack_symmetric_matrix(packed: &[f64], dim: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; dim * dim];
    let mut index = 0;
    for j in 0..dim {
        for i in 0..=j {
            let val = packed[index];
            matrix[j * dim + i] = val; 
            matrix[i * dim + j] = val; 
            index += 1;
        }
    }
    matrix
}

pub fn orthonomalizer(sab: &[f64], dim: usize) -> Vec<f64> {
    let mut packed_sab_copy = sab.to_vec();
    let mut eigval = vec![0.0; dim];
    let mut eigvec = vec![0.0; dim * dim];
    let mut work = vec![0.0; 3*dim];
    let mut info = 0;
    
    unsafe {
        dspev(b'V', b'U', dim as i32, &mut packed_sab_copy, &mut eigval, &mut eigvec, dim as i32, &mut work, &mut info);
    }
    if info != 0 { panic!("Error: info = {}", info); }
    
    let mut x_packed = Vec::with_capacity(dim * (dim + 1) / 2);
    for j in 0..dim {
        for i in 0..=j {
            let mut sum = 0.0;
            for k in 0..dim {
                // Kanonischer Orthogonalisierungs-Filter (1e-4) für lineare Unabhängigkeit!
                if eigval[k] > 1e-4 {
                    sum += eigvec[k * dim + i] * (1.0 / eigval[k].sqrt()) * eigvec[k * dim + j];
                }
            }
            x_packed.push(sum);
        }
    }
    x_packed
}

pub fn get_occupation(n_elec: f64, n_basis: usize, spin_mode:&str) -> Vec<f64> {
    let mut occupation = vec![0.0; n_basis];
    let n_occ = (n_elec / 2.0).floor() as usize;
    for i in 0..n_occ { occupation[i] = 2.0; }
    if spin_mode == "U" && n_elec % 2.0 != 0.0 { occupation[n_occ] = 1.0; }
    occupation
}

pub fn get_rhf_energy(h_core: &[f64], fock_matrix: &[f64], p_matrix: &[f64], n_basis: usize, enuc: f64) -> f64 {
    let mut e_elec = 0.0;
    for i in 0..n_basis {
        for j in 0..n_basis {
            e_elec += 0.5 * p_matrix[j * n_basis + i] * (h_core[j * n_basis + i] + fock_matrix[j * n_basis + i]);
        }
    }
    e_elec + enuc
}

pub fn get_integral_index(i: usize, j: usize) -> usize {
    if i > j { (i * (i + 1)) / 2 + j } else { (j * (j + 1)) / 2 + i }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cargo run <file> <ng> <job_type>");
        process::exit(1);
    }

    let dateiname = &args[1];
    let ng: usize = args[2].parse().unwrap_or(0);
    let job_type = &args[3];

    // =========================================================
    // NEUER BRANCH: Import von fertigen Integralen via PySCF
    // =========================================================
    if job_type == "fcidump" {
        println!("\nReading Integrals directly from FCIDUMP: {}", dateiname);
        let (n_basis, n_elec, e_core, h_mo, eri_mo) = read_fcidump(dateiname);
        
        let n_alpha = n_elec / 2;
        let n_beta = n_elec / 2;
        
        let dets = build_fci_space(n_basis, n_alpha, n_beta);
        println!("FCI Space Dimension: {}", dets.len());

        let (fci_elec_energy, fci_vector) = do_davidson(&dets, &h_mo, &eri_mo, n_basis, 50, 1e-8);
        let total_fci_energy = fci_elec_energy + e_core;

        println!("\n=========================================================");
        println!(" F C I   R E S U L T S   (EXTERNAL FCIDUMP)");
        println!("=========================================================");
        println!("Electronic Energy:     {:.12}", fci_elec_energy);
        println!("Core/Nuclear Energy:   {:.12}", e_core);
        println!("Total FCI Energy:      {:.12}", total_fci_energy);
        println!("=========================================================");

        crate::fci::analyze_fci_vector(&dets, &fci_vector, n_basis, 0.001);
        
        return; 
    }

    let content = fs::read_to_string(&args[1]).expect("File not readable");
    let (sys, flags) = parse_hf_input(&content).expect("Parse failed");

    println!("\nBuilding Static Data...");
    let static_data = StaticHFData::build(&sys, ng);
    let current_xyz: Vec<[f64; 3]> = sys.atoms.iter().map(|a| [a.x, a.y, a.z]).collect();

    if job_type == "sp" || job_type == "mp2" || job_type == "fci" {
        println!("\nStarting Single-Point Energy Calculation...");
        let scf_result = calculate_single_point_energy(&current_xyz, &static_data, None, false);

        println!("\n=========================================================");
        println!(" F I N A L   R E S U L T S");
        println!("=========================================================");
        println!("Total RHF Energy: {:.12} ", scf_result.energy);
        
        if job_type == "mp2"{
            println!("\nStarting MP2 Correction...");
            let mp2_energy = crate::mp2::do_mp2_cycle(&scf_result, &static_data);
            let total_energy = scf_result.energy + mp2_energy; 

            println!("\nMP2 Correlation Energy: {:.12} ", mp2_energy);
            println!("Total MP2 Energy:       {:.12} ", total_energy);
        } else if job_type == "fci" {
            println!("\nStarting FCI Calculation...");
            
            let n_basis = static_data.n_basis;
            let total_electrons: f64 = static_data.z_nuc.iter().sum(); 
            let n_alpha = (total_electrons / 2.0).floor() as usize;
            let n_beta = (total_electrons / 2.0).ceil() as usize;
            let enuc = calc_nuc_rep(&current_xyz, &static_data.z_nuc);
            
            let h_core = &scf_result.h_core;
            let ao_integrals = &scf_result.ao_integrals;

            println!("Transforming Integrals to MO Basis...");
            let h_mo = transform_1e_ao_to_mo(h_core, &scf_result.c_matrix, n_basis); 
            let eri_mo = transform_integrals_ao_to_mo(ao_integrals, &scf_result.c_matrix, n_basis);

            let dets = build_fci_space(n_basis, n_alpha, n_beta);
            println!("FCI Space Dimension: {}", dets.len());

            let (fci_elec_energy, fci_vector) = do_davidson(&dets, &h_mo, &eri_mo, n_basis, 50, 1e-8);

            let total_fci_energy = fci_elec_energy + enuc;
            
            println!("\n=========================================================");
            println!(" F C I   R E S U L T S");
            println!("=========================================================");
            println!("FCI Electronic Energy: {:.12}", fci_elec_energy);
            println!("Nuclear Repulsion:     {:.12} ", enuc);
            println!("Total FCI Energy:      {:.12} ", total_fci_energy);
            println!("=========================================================");

            crate::fci::analyze_fci_vector(&dets, &fci_vector, n_basis, 0.001);
        }

        mulliken_pop_analysis(
            &scf_result.p_matrix, 
            &scf_result.overlap, 
            &static_data.basis_to_atom, 
            static_data.n_basis, 
            &static_data.z_nuc
        );
        
    } else if job_type == "opt" {
        println!("\nStarting Optimization Setup...");
        
        let scf_result = calculate_single_point_energy(&current_xyz, &static_data, None, true);

        if flags.is_empty() {
            println!("No 'OPT=' flags found in input. Defaulting to standard Nuclear Geometry Optimization...");
            
            let optimized_xyz = do_optimization(&current_xyz, &static_data, scf_result.p_matrix, 200, 1e-4);

            println!("\n=========================================================");
            println!(" F I N A L   O P T I M I Z E D   G E O M E T R Y");
            println!("=========================================================");
            for (i, atom) in optimized_xyz.iter().enumerate() {
                println!("Atom {:>2}: X: {:>10.5} | Y: {:>10.5} | Z: {:>10.5}", i + 1, atom[0], atom[1], atom[2]);
            }
        } else {
            println!("\nStarting Master Optimization with active flags: {:?}", flags);
            
            let (_final_xyz, _final_static_data) = do_master_optimization(
                &current_xyz, 
                &static_data, 
                &scf_result.p_matrix, 
                1000, 
                1e-4, 
                flags
            );
            
        }
    } else {
        eprintln!("Unknown job type: {}. Use 'sp', 'mp2', or 'opt'.", job_type);
    }
}