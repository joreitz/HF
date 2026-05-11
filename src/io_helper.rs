use std::fs::File;
use std::io::Write;
use crate::types::StaticHFData; // Pfad anpassen, falls nötig

pub fn export_to_input_format(filename: &str, static_data: &StaticHFData) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    writeln!(file, "{} {} {}", static_data.n_atoms, static_data.n_elec, static_data.n_basis)?;

    for i in 0..static_data.n_atoms {

        let mut basis_funcs_for_atom = Vec::new();
        for mu in 0..static_data.n_basis {
            if static_data.basis_to_atom[mu] == i {
                basis_funcs_for_atom.push(mu);
            }
        }

        let n_basis_atom = basis_funcs_for_atom.len();

 
        if n_basis_atom == 0 {

            writeln!(file, "0.0 0.0 0.0 {:.1} 0", static_data.z_nuc[i])?;
            continue;
        }

        let first_mu = basis_funcs_for_atom[0];
        let x = static_data.centers[first_mu][0];
        let y = static_data.centers[first_mu][1];
        let z = static_data.centers[first_mu][2];
        let z_nuc = static_data.z_nuc[i];


        writeln!(file, "{:.6}  {:.6}  {:.6}  {:.1}  {}", x, y, z, z_nuc, n_basis_atom)?;

  
        for &mu in &basis_funcs_for_atom {
            writeln!(file, "{:.6}", static_data.zetas[mu])?;
        }
    }

    Ok(())
}