/// Module containing quantum chemistry molecular integrals over spherical Gaussian functions.
pub mod integrals {
    use std::f64::consts::PI;

    // In Rust, the Error Function isn't in standard library natively.
    // You will need to add the `libm` crate to your Cargo.toml:
    // [dependencies]
    // libm = "0.2"
    use libm::erf;

    const TPI: f64 = 2.0 * PI;

    /// Computes one-electron integrals over spherical Gaussian functions.
    /// 
    /// Returns a tuple containing `(sab, tab, vab)`:
    /// - `sab`: overlap integral <a|b>
    /// - `tab`: kinetic energy integral <a|T|b>
    /// - `vab`: nuclear attraction integral <a|Σ z/r|b>
    pub fn oneint(
        xyz: &[[f64; 3]],
        chrg: &[f64],
        r_a: &[f64; 3],
        r_b: &[f64; 3],
        alp: &[f64],
        bet: &[f64],
        ci: &[f64],
        cj: &[f64],
    ) -> (f64, f64, f64) {
        let nat = xyz.len().min(chrg.len());
        let npa = alp.len().min(ci.len());
        let npb = bet.len().min(cj.len());

        let mut sab = 0.0;
        let mut tab = 0.0;
        let mut vab = 0.0;

        let rab: f64 = r_a
            .iter()
            .zip(r_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        for i in 0..npa {
            for j in 0..npb {
                let eab = alp[i] + bet[j];
                let oab = 1.0 / eab;
                let cab = ci[i] * cj[j];
                let xab = alp[i] * bet[j] * oab;
                let est = rab * xab;
                let ab = (-est).exp();
                let s00 = cab * ab * (PI * oab).sqrt().powi(3);

                // Overlap
                sab += s00;

                // Kinetic energy
                tab += xab * (3.0 - 2.0 * est) * s00;

                // Nuclear attraction
                let fact = cab * TPI * oab * ab;
                let mut r_p = [0.0; 3];
                for m in 0..3 {
                    r_p[m] = (alp[i] * r_a[m] + bet[j] * r_b[m]) * oab;
                }

                for k in 0..nat {
                    if chrg[k] == 0.0 {
                        continue;
                    }
                    
                    let rcp: f64 = r_p
                        .iter()
                        .zip(xyz[k].iter())
                        .map(|(p, x)| (p - x).powi(2))
                        .sum();
                    vab -= fact * chrg[k] * boysf0(eab * rcp);
                }
            }
        }

        (sab, tab, vab)
    }

    /// Two-electron repulsion integral (ab|cd) over spherical Gaussian functions
    /// Quantity is given in chemist's notation
    pub fn twoint(
        r_a: &[f64; 3],
        r_b: &[f64; 3],
        r_c: &[f64; 3],
        r_d: &[f64; 3],
        alp: &[f64],
        bet: &[f64],
        gam: &[f64],
        del: &[f64],
        ci: &[f64],
        cj: &[f64],
        ck: &[f64],
        cl: &[f64],
    ) -> f64 {
        let npa = alp.len().min(ci.len());
        let npb = bet.len().min(cj.len());
        let npc = gam.len().min(ck.len());
        let npd = del.len().min(cl.len());

        let mut tei = 0.0;
        let twopi25 = 2.0 * PI.powf(2.5);

        // R²(a-b)
        let rab: f64 = r_a
            .iter()
            .zip(r_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        // R²(c-d)
        let rcd: f64 = r_c
            .iter()
            .zip(r_d.iter())
            .map(|(c, d)| (c - d).powi(2))
            .sum();

        for i in 0..npa {
            for j in 0..npb {
                let cab = ci[i] * cj[j];
                let eab = alp[i] + bet[j];
                let oab = 1.0 / eab;
                let est_ab = alp[i] * bet[j] * rab * oab;
                let ab = (-est_ab).exp();

                // New Gaussian at r_p
                let mut r_p = [0.0; 3];
                for m in 0..3 {
                    r_p[m] = (alp[i] * r_a[m] + bet[j] * r_b[m]) * oab;
                }

                for k in 0..npc {
                    for l in 0..npd {
                        let ccd = ck[k] * cl[l];
                        let ecd = gam[k] + del[l];
                        let ocd = 1.0 / ecd;
                        let est_cd = gam[k] * del[l] * rcd * ocd;
                        let cd = (-est_cd).exp();

                        // New Gaussian at r_q
                        let mut r_q = [0.0; 3];
                        for m in 0..3 {
                            r_q[m] = (gam[k] * r_c[m] + del[l] * r_d[m]) * ocd;
                        }

                        let abcd = ab * cd;

                        // Distance between product Gaussians
                        let rpq: f64 = r_p
                            .iter()
                            .zip(r_q.iter())
                            .map(|(p, q)| (p - q).powi(2))
                            .sum();

                        let epq = eab * ecd;
                        let eabcd = eab + ecd;
                        let pq = rpq * epq / eabcd;

                        tei += cab
                            * ccd
                            * abcd
                            * twopi25
                            / (epq * eabcd.sqrt())
                            * boysf0(pq);
                    }
                }
            }
        }

        tei
    }

    /// Zeroth order Boys function
    #[inline]
    pub fn boysf0(arg: f64) -> f64 {
        // Six term Taylor expansion is sufficient for precision of 10e-14,
        // use analytical expression for all other terms
        if arg < 0.05 {
            1.0 - 3.333333333333333e-1 * arg
                + 6.666666666666666e-2 * arg.powi(2)
                - 4.761904761904761e-3 * arg.powi(3)
                + 1.763668430335097e-4 * arg.powi(4)
                - 4.008337341670675e-6 * arg.powi(5)
        } else {
            0.5 * (PI / arg).sqrt() * erf(arg.sqrt())
        }
    }
}

pub fn _do_one_elec_int(
    xyz: &[[f64; 3]],
    chrg: &[f64],
    r_a: &[f64; 3],
    r_b: &[f64; 3],
    alp: &[f64],
    bet: &[f64],
    ci: &[f64],
    cj: &[f64],
) -> (f64, f64, f64) {
    integrals::oneint(xyz, chrg, r_a, r_b, alp, bet, ci, cj)
}