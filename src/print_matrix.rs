use std::io::{self, Write};

pub mod print_matrix {
    use super::*;

    macro_rules! with_writer {
        ($opt_writer:expr, $out:ident => $body:expr) => {
            if let Some($out) = $opt_writer {
                $body
            } else {
                let mut stdout = io::stdout();
                let $out = &mut stdout;
                $body
            }
        };
    }

    /// Prints a 1D vector.
    pub fn _write_vector(
        vector: &[f64],
        name: Option<&str>,
        writer: Option<&mut dyn Write>,
    ) -> io::Result<()> {
        let d = vector.len();

        with_writer!(writer, out => {
            if let Some(n) = name {
                writeln!(out, "\nvector printed: {}", n)?;
            }

            for j in 0..d {
                // (i6)
                write!(out, "{:6}", j + 1)?;
                // (1x, f15.8)
                writeln!(out, " {:15.8}", vector[j])?;
            }

            Ok(())
        })
    }

    /// Prints a 2D matrix stored in a flat 1D slice (assumes Column-Major layout).
    pub fn _write_2d_matrix(
        matrix: &[f64],
        rows: usize,
        cols: usize,
        name: Option<&str>,
        writer: Option<&mut dyn Write>,
        step: Option<usize>,
    ) -> io::Result<()> {
        let istep = step.unwrap_or(5);

        with_writer!(writer, out => {
            if let Some(n) = name {
                writeln!(out, "\nmatrix printed: {}", n)?;
            }

            let mut i = 0;
            while i < cols {
                let l = std::cmp::min(i + istep, cols);

                // Print column headers
                // (/, 6x) -> Newline, then 6 spaces (padding for the row index column)
                write!(out, "\n      ")?;
                for k in i..l {
                    // (6x, i7, 3x) -> total 16 chars per column
                    write!(out, "      {:7}   ", k + 1)?;
                }
                writeln!(out)?;

                // Print row data
                for j in 0..rows {
                    // (i6) -> Row index
                    write!(out, "{:6}", j + 1)?;
                    for k in i..l {
                        // Assuming Column-Major storage (Fortran default)
                        let val = matrix[k * rows + j];
                        // (1x, f15.8) -> total 16 chars per column
                        write!(out, " {:15.8}", val)?;
                    }
                    writeln!(out)?;
                }

                i += istep;
            }

            Ok(())
        })
    }

    /// Prints a packed lower-triangular symmetric matrix.
    pub fn _write_packed_matrix(
        matrix: &[f64],
        name: Option<&str>,
        writer: Option<&mut dyn Write>,
        step: Option<usize>,
    ) -> io::Result<()> {
        // d = (sqrt(8*size + 1) - 1) / 2
        let size = matrix.len() as f64;
        let d = (((8.0 * size + 1.0).sqrt() - 1.0) / 2.0).round() as usize;

        let istep = step.unwrap_or(5);

        with_writer!(writer, out => {
            if let Some(n) = name {
                writeln!(out, "\nmatrix printed: {}", n)?;
            }

            let mut i = 0;
            while i < d {
                let l = std::cmp::min(i + istep, d);

                // Print column headers
                write!(out, "\n      ")?;
                for k in i..l {
                    write!(out, "      {:7}   ", k + 1)?;
                }
                writeln!(out)?;

                // Print row data (lower triangular, so row j starts at i and goes to d)
                for j in i..d {
                    // Limit the printed columns to the diagonal (j + 1) or the block end (l)
                    let end_k = std::cmp::min(l, j + 1);

                    write!(out, "{:6}", j + 1)?;
                    for k in i..end_k {
                        // 0-based packed mapping: (j * (j + 1)) / 2 + k
                        let val = matrix[(j * (j + 1)) / 2 + k];
                        write!(out, " {:15.8}", val)?;
                    }
                    writeln!(out)?;
                }

                i += istep;
            }

            Ok(())
        })
    }
}