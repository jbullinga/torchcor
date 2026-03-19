"""
Compute activation-time (AT) and repolarisation-time (RT) maps from
a cached Vm.pt snapshot tensor and save them as .npy files.

Usage:
    python -m torchcor.ecg._compute_at_rt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

RESULT_DIR = Path(r"C:\Users\bulli\cardiac_data\kcl_torso\KCL_torso3\sim\results")

# Snapshot spacing.  Vm has 601 frames over 600 ms → 1 ms/frame.
DT_SNAPSHOT_MS = 1.0

AT_THRESHOLD_MV  =   0.0   # depolarisation  (upstroke crosses 0 mV)
RT_THRESHOLD_MV  = -70.0   # repolarisation  (falling edge crosses -70 mV)


def _first_crossing_time(Vm: torch.Tensor, threshold: float,
                          rising: bool, dt: float) -> np.ndarray:
    """
    For each node find the first time (ms) the signal crosses `threshold`
    in the specified direction (rising=True → crossing upward).

    Vm   : (T, N) float32 tensor
    Returns (N,) float32 array in ms; NaN where no crossing found.
    """
    T, N = Vm.shape
    # Shift by threshold so crossing = sign change through 0
    above = (Vm > threshold)          # (T, N) bool

    if rising:
        # Was below, now above
        cross = (~above[:-1]) & above[1:]   # (T-1, N)
    else:
        # Was above, now below
        cross = above[:-1] & (~above[1:])   # (T-1, N)

    # For each node: index of first True in the T-1 dimension
    # argmax returns 0 if no True → we'll mask those separately
    first_idx = cross.long().argmax(dim=0)   # (N,)
    has_cross  = cross.any(dim=0)            # (N,)

    # Time: index + 1 (the second sample of the pair that straddles the crossing)
    t_ms = (first_idx.float() + 1.0) * dt
    t_ms[~has_cross] = float("nan")
    return t_ms.cpu().numpy()


def main() -> None:
    vm_path = RESULT_DIR / "Vm.pt"
    if not vm_path.exists():
        raise FileNotFoundError(f"Vm.pt not found at {vm_path}. Run the pipeline first.")

    print(f"Loading {vm_path} …", flush=True)
    Vm = torch.load(str(vm_path), map_location="cpu")
    T, N = Vm.shape
    duration_ms = (T - 1) * DT_SNAPSHOT_MS
    print(f"  Vm shape : {Vm.shape}   ({duration_ms:.0f} ms)")

    print("Computing activation times (rising, threshold=0 mV) …", flush=True)
    AT = _first_crossing_time(Vm, AT_THRESHOLD_MV, rising=True,  dt=DT_SNAPSHOT_MS)
    print(f"  AT  range : {np.nanmin(AT):.1f} – {np.nanmax(AT):.1f} ms"
          f"   (NaN in {np.isnan(AT).sum()} / {N} nodes)")

    print("Computing repolarisation times (falling, threshold=-70 mV) …", flush=True)
    RT = _first_crossing_time(Vm, RT_THRESHOLD_MV, rising=False, dt=DT_SNAPSHOT_MS)
    print(f"  RT  range : {np.nanmin(RT):.1f} – {np.nanmax(RT):.1f} ms"
          f"   (NaN in {np.isnan(RT).sum()} / {N} nodes)")

    at_path = RESULT_DIR / "activation_times.npy"
    rt_path = RESULT_DIR / "repolarization_times.npy"
    np.save(str(at_path), AT)
    np.save(str(rt_path), RT)
    print(f"\nSaved → {at_path}")
    print(f"Saved → {rt_path}")


if __name__ == "__main__":
    main()
