use crate::types::{System, Atom};
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct OptFlags: u32 {
        const NONE         = 0b00000000;
        const GEOMETRY     = 0b00000001;
        const FSGO_CENTERS = 0b00000010;
        const ZETAS        = 0b00000100;
        const GAUSSIANS    = 0b00001000;
    }
}

pub fn parse_next<'a>(iter: &mut std::str::SplitWhitespace<'a>, field_name: &str) -> Result<f64, String> {
    let token = iter
        .next()
        .ok_or_else(|| format!("Unwanted format: {}", field_name))?;
    
    token.parse::<f64>()
        .map_err(|_| format!("Error parsing {}: '{}' is not a valid f64", field_name, token))
}

pub fn parse_hf_input(content: &str) -> Result<(System, OptFlags), String> {
    let mut tokens = content.split_whitespace();


    let mut first_token = tokens.next().ok_or("Empty input")?;
    let mut opt_flags = OptFlags::NONE;

    if first_token.starts_with("OPT=") {
        if first_token.contains("GEOM")  { opt_flags |= OptFlags::GEOMETRY; }
        if first_token.contains("FSGO")  { opt_flags |= OptFlags::FSGO_CENTERS; }
        if first_token.contains("GAUSS") { opt_flags |= OptFlags::GAUSSIANS; }


        first_token = tokens.next().ok_or("Unwanted format: N_atoms")?;
    }

    let n_atoms = first_token.parse::<f64>()
        .map_err(|_| format!("Error parsing N_atoms: '{}' is not a valid f64", first_token))?;

    let n_elec = parse_next(&mut tokens, "N_elec")?;
    let n_basis = parse_next(&mut tokens, "N_basis")?;

    let mut atoms = Vec::with_capacity(n_atoms as usize);

    for _i in 0..(n_atoms as usize) {
        let x = parse_next(&mut tokens, "X")?;
        let y = parse_next(&mut tokens, "Y")?;
        let z = parse_next(&mut tokens, "Z")?;

        let z_nuc = parse_next(&mut tokens, "Z_nuc")?;
        let n_basis_atom = parse_next(&mut tokens, "N_basis_atom")?;

        let mut zetas = Vec::with_capacity(n_basis_atom as usize);

        for _ in 0..(n_basis_atom as usize) {
            zetas.push(parse_next(&mut tokens, "Zeta")?);
        }

        atoms.push(Atom {
            x, y, z, z_nuc,
            n_basis: n_basis_atom,
            zetas,
        });
    }

    Ok((System { n_atoms, n_elec, n_basis, atoms }, opt_flags))
}