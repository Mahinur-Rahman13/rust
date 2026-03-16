use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

fn idx(i: usize, j: usize, k: usize, n: usize) -> usize {
    i + n * (j + n * k)
}

fn coord(i: usize, n: usize, l: f64) -> f64 {
    -l + (2.0 * l) * (i as f64) / ((n - 1) as f64)
}

fn analytic_potential(x: f64, y: f64, z: f64, q: f64, eps0: f64) -> f64 {
    let r = (x * x + y * y + z * z).sqrt();
    if r == 0.0 {
        f64::INFINITY
    } else {
        q / (4.0 * PI * eps0 * r)
    }
}

fn main() -> std::io::Result<()> {
    // -------------------------
    // Parameters
    // -------------------------
    let n: usize = 65;          // grid points per direction
    let l: f64 = 1.0;           // half-width of domain: [-L, L]
    let eps0: f64 = 1.0;        // choose units so eps0 = 1
    let q: f64 = 1.0;           // point charge
    let omega: f64 = 1.85;      // SOR relaxation factor
    let tol: f64 = 1e-8;
    let max_iter: usize = 50_000;

    let h = 2.0 * l / ((n - 1) as f64);
    let center = n / 2;

    // -------------------------
    // Storage
    // -------------------------
    let size = n * n * n;
    let mut v = vec![0.0_f64; size];
    let mut rho = vec![0.0_f64; size];

    // Put the total charge q into the center cell:
    // rho * h^3 = q  => rho = q / h^3
    rho[idx(center, center, center, n)] = q / (h * h * h);

    // -------------------------
    // Boundary conditions:
    // Use exact Coulomb potential on the six faces
    // -------------------------
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1 {
                    let x = coord(i, n, l);
                    let y = coord(j, n, l);
                    let z = coord(k, n, l);
                    v[idx(i, j, k, n)] = analytic_potential(x, y, z, q, eps0);
                }
            }
        }
    }

    // Optional initial guess for interior: use analytical value too
    // This accelerates convergence and makes comparison cleaner.
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let x = coord(i, n, l);
                let y = coord(j, n, l);
                let z = coord(k, n, l);
                if i == center && j == center && k == center {
                    v[idx(i, j, k, n)] = analytic_potential(h / 2.0, 0.0, 0.0, q, eps0);
                } else {
                    v[idx(i, j, k, n)] = analytic_potential(x, y, z, q, eps0);
                }
            }
        }
    }

    // -------------------------
    // SOR iteration
    // Discrete equation:
    // (sum neighbors - 6V)/h^2 = -rho/eps0
    // => V = (sum neighbors + h^2 rho/eps0)/6
    // -------------------------
    let mut iter = 0usize;
    let mut max_diff;

    loop {
        iter += 1;
        max_diff = 0.0;

        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let p = idx(i, j, k, n);

                    let v_old = v[p];

                    let sum_nb =
                        v[idx(i + 1, j, k, n)] +
                        v[idx(i - 1, j, k, n)] +
                        v[idx(i, j + 1, k, n)] +
                        v[idx(i, j - 1, k, n)] +
                        v[idx(i, j, k + 1, n)] +
                        v[idx(i, j, k - 1, n)];

                    let gs = (sum_nb + h * h * rho[p] / eps0) / 6.0;
                    let v_new = (1.0 - omega) * v_old + omega * gs;

                    let diff = (v_new - v_old).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }

                    v[p] = v_new;
                }
            }
        }

        if iter % 100 == 0 {
            println!("iter = {:6}, max_diff = {:.3e}", iter, max_diff);
        }

        if max_diff < tol || iter >= max_iter {
            break;
        }
    }

    println!("\nFinished.");
    println!("Iterations   = {}", iter);
    println!("Final maxdiff = {:.6e}", max_diff);
    println!("Grid spacing h = {:.6}", h);

    // -------------------------
    // Export z = 0 slice
    // -------------------------
    let kz = center;
    let slice_file = File::create("slice_z0.csv")?;
    let mut slice_writer = BufWriter::new(slice_file);
    writeln!(slice_writer, "x,y,V_num,V_ana")?;

    for j in 0..n {
        for i in 0..n {
            let x = coord(i, n, l);
            let y = coord(j, n, l);
            let z = 0.0;
            let v_num = v[idx(i, j, kz, n)];
            let r = (x * x + y * y + z * z).sqrt();
            let v_ana = if r == 0.0 {
                f64::NAN
            } else {
                analytic_potential(x, y, z, q, eps0)
            };
            writeln!(slice_writer, "{:.12},{:.12},{:.12},{:.12}", x, y, v_num, v_ana)?;
        }
    }

    // -------------------------
    // Export x-axis comparison (y=0, z=0)
    // Avoid the center point because analytic solution is singular there
    // -------------------------
    let cmp_file = File::create("xaxis_compare.csv")?;
    let mut cmp_writer = BufWriter::new(cmp_file);
    writeln!(cmp_writer, "x,V_num,V_ana,rel_err")?;

    let mut max_rel_err = 0.0;
    let mut rms_rel_accum = 0.0;
    let mut count = 0usize;

    for i in 0..n {
        if i == center {
            continue;
        }

        let x = coord(i, n, l);
        let y = 0.0;
        let z = 0.0;

        let v_num = v[idx(i, center, center, n)];
        let v_ana = analytic_potential(x, y, z, q, eps0);

        // Skip points too close to the singularity, where cell-smearing dominates
        if x.abs() < 2.5 * h {
            continue;
        }

        let rel_err = ((v_num - v_ana) / v_ana).abs();
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
        rms_rel_accum += rel_err * rel_err;
        count += 1;

        writeln!(
            cmp_writer,
            "{:.12},{:.12},{:.12},{:.12}",
            x, v_num, v_ana, rel_err
        )?;
    }

    let rms_rel_err = if count > 0 {
        (rms_rel_accum / count as f64).sqrt()
    } else {
        0.0
    };

    println!("\nComparison away from center:");
    println!("sample count       = {}", count);
    println!("max relative error = {:.6e}", max_rel_err);
    println!("rms relative error = {:.6e}", rms_rel_err);

    Ok(())
}