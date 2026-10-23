//! X-Trans demosaic benchmark: re-mosaic linear RGB scenes in simulated camera space, demosaic, score.
//!
//! Scenes are `.rgbf` files (u32 width, u32 height, then width × height × 3 little-endian f32, linear light,
//! white balanced). For each illuminant the scene is divided by that illuminant's white-balance gains (the camera
//! sees less red and blue than the balanced image shows), mosaiced with the X-Trans pattern, optionally given
//! signal-dependent Gaussian noise, demosaiced by each variant, white balanced back and compared with the clean
//! scene after sRGB encoding:
//!
//! - CPSNR (dB, all three channels, 8 px border excluded): overall fidelity.
//! - chroma RMSE (8-bit levels of R − G and B − G): false colour and fringing, which CPSNR mixes with luminance.
//!
//! `cargo run --release -p lightcraft-raw --example xtrans_bench -- <scene dir> [csv out] [dump dir]`

use lightcraft_raw::demosaic::{Options, demosaic_with, mosaic_from_rgb};
use lightcraft_raw::{Cfa, Method, Rgb32f};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const BORDER: usize = 8;

fn load(p: &Path) -> Rgb32f {
    let b = std::fs::read(p).unwrap();
    let w = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let f: Vec<f32> = b[8..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(f.len(), w * h * 3, "{}", p.display());
    Rgb32f { width: w, height: h, data: f.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() }
}

fn save(p: &Path, img: &Rgb32f) {
    let mut b = Vec::with_capacity(8 + img.data.len() * 12);
    b.extend((img.width as u32).to_le_bytes());
    b.extend((img.height as u32).to_le_bytes());
    for px in &img.data {
        for v in px {
            b.extend(v.to_le_bytes());
        }
    }
    std::fs::write(p, b).unwrap();
}

fn srgb(v: f32) -> f64 {
    let v = v.clamp(0.0, 1.0) as f64;
    if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// (CPSNR dB, chroma RMSE in 8-bit levels), both on sRGB-encoded values.
fn score(out: &Rgb32f, truth: &Rgb32f) -> (f64, f64) {
    let (mut se, mut ce, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in BORDER..truth.height - BORDER {
        for x in BORDER..truth.width - BORDER {
            let (p, q) = (out.get(x, y), truth.get(x, y));
            let p = p.map(srgb);
            let q = q.map(srgb);
            for c in 0..3 {
                se += (p[c] - q[c]).powi(2);
            }
            ce += ((p[0] - p[1]) - (q[0] - q[1])).powi(2) + ((p[2] - p[1]) - (q[2] - q[1])).powi(2);
            n += 1;
        }
    }
    (10.0 * (3.0 * n as f64 / se).log10(), 255.0 * (ce / (2 * n) as f64).sqrt())
}

/// Deterministic Gaussian (Box–Muller over an LCG).
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

struct Variant {
    name: &'static str,
    method: Method,
    wb: bool,
    median: bool,
}

const VARIANTS: [Variant; 5] = [
    Variant { name: "bilinear", method: Method::Bilinear, wb: false, median: false },
    Variant { name: "current", method: Method::Ahd, wb: false, median: false },
    Variant { name: "+median", method: Method::Ahd, wb: false, median: true },
    Variant { name: "+wb", method: Method::Ahd, wb: true, median: false },
    Variant { name: "+wb+median", method: Method::Ahd, wb: true, median: true },
];

/// White-balance gains (camera → balanced, green = 1), roughly those of an X-Trans body.
const ILLUMINANTS: [(&str, [f32; 3]); 3] = [("neutral", [1.0, 1.0, 1.0]), ("daylight", [1.9, 1.0, 1.6]), ("tungsten", [1.15, 1.0, 2.8])];

/// Noise variance = a · signal + b (camera units, white = 1).
const NOISE: [(&str, f32, f32); 3] = [("clean", 0.0, 0.0), ("iso800", 1.0e-4, 1.0e-6), ("iso6400", 1.6e-3, 2.5e-5)];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("bench_data"));
    let csv_path = args.get(2);
    let dump = args.get(3).map(Path::new);
    let mut scenes: Vec<_> =
        std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "rgbf")).collect();
    scenes.sort();
    let cfa = Cfa::xtrans();
    let mut csv = String::from("scene,illuminant,noise,variant,cpsnr_db,chroma_rmse,ms\n");
    // (illuminant, noise, variant) → sums
    let mut tot = vec![[0.0f64; 3]; ILLUMINANTS.len() * NOISE.len() * VARIANTS.len()];
    for path in &scenes {
        let scene = path.file_stem().unwrap().to_string_lossy().to_string();
        let truth = load(path);
        for (ii, (iname, gain)) in ILLUMINANTS.iter().enumerate() {
            let cam = Rgb32f {
                width: truth.width,
                height: truth.height,
                data: truth.data.iter().map(|p| [p[0] / gain[0], p[1] / gain[1], p[2] / gain[2]]).collect(),
            };
            for (ni, (nname, a, b)) in NOISE.iter().enumerate() {
                let mut n = mosaic_from_rgb(&cam, &cfa);
                if *a > 0.0 || *b > 0.0 {
                    let mut rng = Rng(0x5eed ^ (ii as u64) << 8 ^ ni as u64);
                    for v in &mut n.data {
                        *v += (a * v.max(0.0) + b).sqrt() * rng.gauss();
                    }
                }
                for (vi, v) in VARIANTS.iter().enumerate() {
                    let opts = Options { balance: v.wb.then_some(*gain), chroma_median: v.median };
                    let t0 = Instant::now();
                    let out = demosaic_with(&n, v.method, opts);
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    let shown = Rgb32f {
                        width: out.width,
                        height: out.height,
                        data: out.data.iter().map(|p| [p[0] * gain[0], p[1] * gain[1], p[2] * gain[2]]).collect(),
                    };
                    let (ps, ch) = score(&shown, &truth);
                    csv.push_str(&format!("{scene},{iname},{nname},{},{ps:.3},{ch:.3},{ms:.1}\n", v.name));
                    let t = &mut tot[(ii * NOISE.len() + ni) * VARIANTS.len() + vi];
                    t[0] += ps;
                    t[1] += ch;
                    t[2] += ms;
                    if let Some(d) = dump
                        && ni == 0
                        && *iname != "neutral"
                    {
                        std::fs::create_dir_all(d).unwrap();
                        save(&d.join(format!("{scene}.{iname}.{}.rgbf", v.name.replace('+', "p"))), &shown);
                    }
                }
            }
        }
        eprintln!("done {scene}");
    }
    let k = scenes.len() as f64;
    println!("Mean over {} scenes (CPSNR dB ↑ / chroma RMSE ↓, sRGB 8-bit levels)\n", scenes.len());
    print!("{:<10} {:<8}", "light", "noise");
    for v in &VARIANTS {
        print!(" | {:>15}", v.name);
    }
    println!();
    for (ii, (iname, _)) in ILLUMINANTS.iter().enumerate() {
        for (ni, (nname, _, _)) in NOISE.iter().enumerate() {
            print!("{iname:<10} {nname:<8}");
            for vi in 0..VARIANTS.len() {
                let t = tot[(ii * NOISE.len() + ni) * VARIANTS.len() + vi];
                print!(" | {:>6.2} / {:>6.2}", t[0] / k, t[1] / k);
            }
            println!();
        }
    }
    if let Some(p) = csv_path {
        std::fs::File::create(p).unwrap().write_all(csv.as_bytes()).unwrap();
    }
}
