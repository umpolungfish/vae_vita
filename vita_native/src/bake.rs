//! vita-bake — flatten a trained lattice checkpoint into the raw blob the
//! mOMonadOS mouth loads. No safetensors/serde on the OS side: each tensor is
//! written as (u32 name_len, name bytes, u32 ndim, u32 dims…, f32 LE data),
//! names sorted, magic "VITA" + u32 tensor count up front.
//!
//! Usage: vita-bake <trunk.safetensors> [out.bin]

use std::io::Write as _;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = args.get(1).cloned().unwrap_or_else(|| "vita_lattice_tri.safetensors".into());
    let out_path = args.get(2).cloned().unwrap_or_else(|| "vita_weights.bin".into());

    let tensors = candle_core::safetensors::load(&src, &candle_core::Device::Cpu)
        .expect("load safetensors");
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let mut out = Vec::new();
    out.extend_from_slice(b"VITA");
    out.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for name in &names {
        let t = &tensors[name.as_str()];
        let dims = t.dims().to_vec();
        let data: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in &dims {
            out.extend_from_slice(&(*d as u32).to_le_bytes());
        }
        for v in &data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut f = std::fs::File::create(&out_path).expect("create out");
    f.write_all(&out).expect("write blob");
    eprintln!(
        "[vita-bake] {} tensors from {src} → {out_path} ({} bytes)",
        names.len(),
        out.len()
    );
}
