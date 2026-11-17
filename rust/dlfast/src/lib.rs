//! Rust hot paths for the DL final project.
//!
//! Every function here has a NumPy reference implementation in
//! `dl_final_prj.fast` and a test that compares the two. Python code should call
//! the wrappers in `dl_final_prj.fast`, not this module directly.

use numpy::ndarray::{Array2, Array4, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArray4, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Semicolon table with decimal commas (UCI LD2011_2014.txt)
// ---------------------------------------------------------------------------

fn unquote(s: &str) -> &str {
    s.trim().trim_matches('"')
}

fn parse_decimal_comma(field: &str) -> Result<f64, String> {
    let f = unquote(field);
    if f.is_empty() {
        return Ok(f64::NAN);
    }
    if f.contains(',') {
        f.replacen(',', ".", 1).parse::<f64>()
    } else {
        f.parse::<f64>()
    }
    .map_err(|e| format!("cannot parse {f:?}: {e}"))
}

/// Parse a `;`-separated table whose first column is a quoted timestamp and whose
/// other columns use a decimal comma. Returns (column names, timestamps, values).
#[pyfunction]
fn parse_semicolon_table<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<(Vec<String>, Vec<String>, Bound<'py, PyArray2<f64>>)> {
    let text = std::fs::read_to_string(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    let result = py.detach(|| -> Result<(Vec<String>, Vec<String>, Array2<f64>), String> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or("empty file")?;
        let names: Vec<String> = header.split(';').skip(1).map(|s| unquote(s).to_string()).collect();
        let k = names.len();
        let rows: Vec<&str> = lines.collect();
        let parsed: Vec<Result<(String, Vec<f64>), String>> = rows
            .par_iter()
            .enumerate()
            .map(|(i, line)| {
                let mut fields = line.split(';');
                let stamp = unquote(fields.next().unwrap_or("")).to_string();
                let vals: Vec<f64> = fields.map(parse_decimal_comma).collect::<Result<_, _>>()?;
                if vals.len() != k {
                    return Err(format!("row {} has {} values, expected {}", i + 2, vals.len(), k));
                }
                Ok((stamp, vals))
            })
            .collect();
        let mut stamps = Vec::with_capacity(rows.len());
        let mut flat = Vec::with_capacity(rows.len() * k);
        for r in parsed {
            let (s, v) = r?;
            stamps.push(s);
            flat.extend_from_slice(&v);
        }
        let arr = Array2::from_shape_vec((stamps.len(), k), flat).map_err(|e| e.to_string())?;
        Ok((names, stamps, arr))
    });
    let (names, stamps, arr) = result.map_err(PyValueError::new_err)?;
    Ok((names, stamps, arr.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Image letterboxing (SVHN cache)
// ---------------------------------------------------------------------------

struct Geometry {
    orig_w: f32,
    orig_h: f32,
    scale: f32,
    x_off: f32,
    y_off: f32,
}

fn letterbox_one(
    path: &str,
    crop: Option<[f32; 4]>,
    height: u32,
    width: u32,
    pad: [u8; 3],
    stretch: bool,
    out: &mut [u8],
) -> Result<Geometry, String> {
    let img = image::open(path).map_err(|e| format!("{path}: {e}"))?.to_rgb8();
    let (w0, h0) = img.dimensions();
    let img = match crop {
        Some([l, t, r, b]) => {
            let l = l.floor().clamp(0.0, (w0 - 1) as f32) as u32;
            let t = t.floor().clamp(0.0, (h0 - 1) as f32) as u32;
            let r = r.ceil().clamp((l + 1) as f32, w0 as f32) as u32;
            let b = b.ceil().clamp((t + 1) as f32, h0 as f32) as u32;
            image::imageops::crop_imm(&img, l, t, r - l, b - t).to_image()
        }
        None => img,
    };
    let (w, h) = img.dimensions();
    if stretch {
        // Fill the whole canvas, ignoring the aspect ratio. Scale is the horizontal factor.
        let resized = image::imageops::resize(&img, width, height, image::imageops::FilterType::Triangle);
        out.copy_from_slice(resized.as_raw());
        return Ok(Geometry { orig_w: w as f32, orig_h: h as f32, scale: width as f32 / w as f32, x_off: 0.0, y_off: 0.0 });
    }
    let scale = (height as f32 / h as f32).min(width as f32 / w as f32);
    let nw = ((w as f32 * scale).round() as u32).clamp(1, width);
    let nh = ((h as f32 * scale).round() as u32).clamp(1, height);
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    let x_off = (width - nw) / 2;
    let y_off = (height - nh) / 2;
    for px in out.chunks_exact_mut(3) {
        px.copy_from_slice(&pad);
    }
    let row_bytes = (width * 3) as usize;
    for y in 0..nh {
        let dst = ((y + y_off) as usize) * row_bytes + (x_off as usize) * 3;
        let src_row = &resized.as_raw()[(y * nw * 3) as usize..((y + 1) * nw * 3) as usize];
        out[dst..dst + src_row.len()].copy_from_slice(src_row);
    }
    Ok(Geometry { orig_w: w as f32, orig_h: h as f32, scale, x_off: x_off as f32, y_off: y_off as f32 })
}

/// Decode, optionally crop, resize with preserved aspect ratio, and center-pad each image
/// into a (height, width, 3) uint8 canvas. Returns images (N, H, W, 3) and geometry
/// (N, 5) with columns [source_w, source_h, scale, x_offset, y_offset], where source is
/// the image after the optional crop. With `stretch`, the source is resized to exactly
/// (width, height) ignoring its aspect ratio, so the offsets are 0 and scale is width/source_w.
#[pyfunction]
#[pyo3(signature = (paths, height, width, pad, crops=None, *, stretch=false))]
fn letterbox_images<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    height: u32,
    width: u32,
    pad: [u8; 3],
    crops: Option<PyReadonlyArray2<'py, f32>>,
    stretch: bool,
) -> PyResult<(Bound<'py, PyArray4<u8>>, Bound<'py, PyArray2<f32>>)> {
    let n = paths.len();
    let crops: Option<Vec<[f32; 4]>> = match crops {
        Some(c) => {
            let v = c.as_array();
            if v.shape() != [n, 4] {
                return Err(PyValueError::new_err("crops must have shape (N, 4)"));
            }
            Some(v.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3]]).collect())
        }
        None => None,
    };
    let per = (height * width * 3) as usize;
    let result = py.detach(|| -> Result<(Vec<u8>, Vec<f32>), String> {
        let mut buf = vec![0u8; n * per];
        let geoms: Vec<Result<Geometry, String>> = buf
            .par_chunks_mut(per)
            .enumerate()
            .map(|(i, out)| {
                letterbox_one(&paths[i], crops.as_ref().map(|c| c[i]), height, width, pad, stretch, out)
            })
            .collect();
        let mut g = Vec::with_capacity(n * 5);
        for r in geoms {
            let x = r?;
            g.extend_from_slice(&[x.orig_w, x.orig_h, x.scale, x.x_off, x.y_off]);
        }
        Ok((buf, g))
    });
    let (buf, g) = result.map_err(PyIOError::new_err)?;
    let imgs = Array4::from_shape_vec((n, height as usize, width as usize, 3), buf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let geom = Array2::from_shape_vec((n, 5), g).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((imgs.into_pyarray(py), geom.into_pyarray(py)))
}

/// Per-channel RGB mean over all pixels of all images, in [0, 255].
#[pyfunction]
fn mean_rgb(py: Python<'_>, paths: Vec<String>) -> PyResult<[f64; 3]> {
    let result = py.detach(|| -> Result<[f64; 3], String> {
        let parts: Vec<Result<([f64; 3], f64), String>> = paths
            .par_iter()
            .map(|p| {
                let img = image::open(p).map_err(|e| format!("{p}: {e}"))?.to_rgb8();
                let mut s = [0f64; 3];
                for px in img.pixels() {
                    for c in 0..3 {
                        s[c] += px[c] as f64;
                    }
                }
                Ok((s, (img.width() * img.height()) as f64))
            })
            .collect();
        let mut total = [0f64; 3];
        let mut count = 0f64;
        for r in parts {
            let (s, c) = r?;
            for k in 0..3 {
                total[k] += s[k];
            }
            count += c;
        }
        if count == 0.0 {
            return Err("no pixels".into());
        }
        Ok([total[0] / count, total[1] / count, total[2] / count])
    });
    result.map_err(PyIOError::new_err)
}

// ---------------------------------------------------------------------------
// Moving-block bootstrap
// ---------------------------------------------------------------------------

/// SplitMix64. Small, fast and deterministic, so results do not depend on thread count.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform integer in [0, m).
    fn below(&mut self, m: u64) -> u64 {
        ((self.next_u64() as u128 * m as u128) >> 64) as u64
    }
}

fn prefix_sums(x: &ArrayView2<f64>) -> Vec<f64> {
    let (n, k) = x.dim();
    let mut cs = vec![0f64; (n + 1) * k];
    for i in 0..n {
        for j in 0..k {
            cs[(i + 1) * k + j] = cs[i * k + j] + x[[i, j]];
        }
    }
    cs
}

/// Column means of one resample made of blocks starting at `starts`. Blocks have length
/// `block`, the last one is truncated so the resample has exactly n rows.
fn resample_means(cs: &[f64], n: usize, k: usize, block: usize, starts: &[usize], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    let mut remaining = n;
    for &s in starts {
        let len = block.min(remaining);
        for j in 0..k {
            out[j] += cs[(s + len) * k + j] - cs[s * k + j];
        }
        remaining -= len;
        if remaining == 0 {
            break;
        }
    }
    out.iter_mut().for_each(|v| *v /= n as f64);
}

fn check_block(n: usize, block: usize) -> PyResult<usize> {
    if n == 0 || block == 0 || block > n {
        return Err(PyValueError::new_err("need 1 <= block <= n and n > 0"));
    }
    Ok(n.div_ceil(block))
}

/// Moving-block bootstrap of column means. `columns` is (n, k), for example squared and
/// absolute errors of two models at each test origin. Returns (reps, k) resampled means.
/// Each replicate draws ceil(n/block) block starts uniformly from [0, n-block].
#[pyfunction]
fn block_bootstrap_means<'py>(
    py: Python<'py>,
    columns: PyReadonlyArray2<'py, f64>,
    block: usize,
    reps: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = columns.as_array();
    let (n, k) = x.dim();
    let nblocks = check_block(n, block)?;
    let cs = prefix_sums(&x);
    let max_start = (n - block + 1) as u64;
    let out = py.detach(|| {
        let mut out = vec![0f64; reps * k];
        out.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
            let mut rng = SplitMix64(seed ^ (r as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let starts: Vec<usize> = (0..nblocks).map(|_| rng.below(max_start) as usize).collect();
            resample_means(&cs, n, k, block, &starts, row);
        });
        out
    });
    let arr = Array2::from_shape_vec((reps, k), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Deterministic core of `block_bootstrap_means` with given block starts (reps, nblocks).
/// Used to test the Rust result against NumPy.
#[pyfunction]
fn block_means_from_starts<'py>(
    py: Python<'py>,
    columns: PyReadonlyArray2<'py, f64>,
    block: usize,
    starts: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = columns.as_array();
    let (n, k) = x.dim();
    let nblocks = check_block(n, block)?;
    let st = starts.as_array();
    if st.ncols() != nblocks {
        return Err(PyValueError::new_err(format!("starts needs {nblocks} columns")));
    }
    if st.iter().any(|&s| s < 0 || s as usize + block > n) {
        return Err(PyValueError::new_err("block start out of range"));
    }
    let cs = prefix_sums(&x);
    let reps = st.nrows();
    let mut out = vec![0f64; reps * k];
    for (r, row) in out.chunks_mut(k).enumerate() {
        let s: Vec<usize> = st.row(r).iter().map(|&v| v as usize).collect();
        resample_means(&cs, n, k, block, &s, row);
    }
    let arr = Array2::from_shape_vec((reps, k), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Edit distance
// ---------------------------------------------------------------------------

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Levenshtein distance for each pair (a[i], b[i]), on Unicode characters.
#[pyfunction]
fn levenshtein_batch(py: Python<'_>, a: Vec<String>, b: Vec<String>) -> PyResult<Vec<usize>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("a and b must have the same length"));
    }
    Ok(py.detach(|| {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| {
                let xc: Vec<char> = x.chars().collect();
                let yc: Vec<char> = y.chars().collect();
                levenshtein(&xc, &yc)
            })
            .collect()
    }))
}

#[pymodule]
fn dlfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_semicolon_table, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_images, m)?)?;
    m.add_function(wrap_pyfunction!(mean_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(block_bootstrap_means, m)?)?;
    m.add_function(wrap_pyfunction!(block_means_from_starts, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_batch, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        let c = |s: &str| s.chars().collect::<Vec<_>>();
        assert_eq!(levenshtein(&c("kitten"), &c("sitting")), 3);
        assert_eq!(levenshtein(&c(""), &c("12")), 2);
        assert_eq!(levenshtein(&c("123"), &c("123")), 0);
    }

    #[test]
    fn decimal_comma() {
        assert_eq!(parse_decimal_comma("2,5").unwrap(), 2.5);
        assert_eq!(parse_decimal_comma("0").unwrap(), 0.0);
        assert!(parse_decimal_comma("\"\"").unwrap().is_nan());
    }

    #[test]
    fn letterbox_stretch_fills_canvas() {
        let path = std::env::temp_dir().join(format!("dlfast_stretch_{}.png", std::process::id()));
        let img = image::RgbImage::from_fn(20, 10, |x, y| image::Rgb([(x * 10) as u8, (y * 20) as u8, 7]));
        img.save(&path).unwrap();
        let mut out = vec![0u8; 8 * 6 * 3];
        let p = path.to_str().unwrap();
        let g = letterbox_one(p, Some([2.0, 1.0, 12.0, 9.0]), 8, 6, [1, 2, 3], true, &mut out).unwrap();
        assert_eq!((g.orig_w, g.orig_h, g.x_off, g.y_off), (10.0, 8.0, 0.0, 0.0));
        assert!((g.scale - 0.6).abs() < 1e-6);
        assert!(out.chunks_exact(3).all(|px| px[2] == 7)); // no padding pixel anywhere
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rng_below_in_range() {
        let mut r = SplitMix64(7);
        assert!((0..10_000).all(|_| r.below(13) < 13));
    }
}
