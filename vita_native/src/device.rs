//! Which card the run uses, from `IG_DEVICES` — the one spelling for device
//! selection across every repo here.
//!
//! Two cards in this box are two separate memories, not one pool. A trunk this
//! size fits either alone, so the choice here is WHICH card rather than how to
//! split one model across both: name an ordinal (`IG_DEVICES=1`), name several
//! and the first that opens is used, `cpu` to force the CPU, or leave it unset
//! and the first card that opens takes the run. Two trainings pinned to two
//! ordinals is the way both cards work at once here — one model each, full VRAM
//! each, no hidden state crossing between them.

use candle_core::Device;

/// Parse `IG_DEVICES` and open the first ordinal that answers.
pub fn pick() -> Device {
    // Match candle's ordinals to nvidia-smi's, so IG_DEVICES=1 means the card
    // the user sees as 1 rather than whichever the driver ranks second.
    if std::env::var("CUDA_DEVICE_ORDER").is_err() {
        std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    }
    let spec = std::env::var("IG_DEVICES").unwrap_or_default().trim().to_ascii_lowercase();
    if spec == "cpu" || std::env::var("IG_LOCAL_CPU").is_ok() {
        return Device::Cpu;
    }
    let wanted: Vec<usize> = if spec.is_empty() || spec == "auto" || spec == "all" {
        (0..8).collect()
    } else {
        spec.split(',').filter_map(|p| p.trim().parse().ok()).collect()
    };
    for idx in wanted {
        if let Ok(d) = Device::new_cuda(idx) {
            return d;
        }
    }
    Device::Cpu
}
